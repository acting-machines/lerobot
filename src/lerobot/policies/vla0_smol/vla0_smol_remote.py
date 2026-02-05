import asyncio
import base64
import functools
import io
import logging
from collections.abc import AsyncGenerator

import numpy as np
import requests
import torch
from openai import AsyncOpenAI
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import CenterCrop

from lerobot.policies.vla0_smol.vla0_smol_common import EPS, build_exact_n_numbers_grammar
from lerobot.utils.constants import OBS_STATE

logging.getLogger("httpx").setLevel(logging.WARNING)


SERVER_PROFILE = False


class VLA0Client(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]
        self.image_keys = self.config.image_features.keys()
        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.center_crop_fn = CenterCrop(config.crop_shape)

        self.model_name = "vla-0-smol"

        total_actions = self.config.chunk_size * self.config.action_feature.shape[0]
        self.grammar_str = build_exact_n_numbers_grammar(total_actions, 0, self.config.n_action_bins)

        # Dummy param for device management
        self.register_buffer("dummy_param", torch.empty(0))

    def forward(self, batch):
        raise NotImplementedError("Client backend cannot be trained. Use use_remote_client=False.")

    def _process_image_to_base64(self, tensor_img: Tensor, format="PNG") -> str:
        """
        Converts a (C, H, W) float tensor to a Base64 encoded string.
        """
        if self.do_crop:
            tensor_img = self.center_crop_fn(tensor_img)

        arr = tensor_img.permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).astype(np.uint8)

        pil_img = Image.fromarray(arr)
        buff = io.BytesIO()
        if format == "PNG":
            # TODO: Check if optimize=True helps reduce size without quality loss
            pil_img.save(buff, format="PNG", optimize=True)
        else:
            pil_img.save(buff, format="JPEG", quality=95)
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    async def _build_prompt_payload(self, batch, index):
        loop = asyncio.get_running_loop()

        state = batch[OBS_STATE][index]
        prompt_text = batch["prefix"][index]

        content_payload = []
        present_img_keys = [k for k in self.image_keys if k in batch]

        for key in present_img_keys:
            img_tensor = batch[key][index]
            # functools.partial is needed to pass args to run_in_executor
            b64_str = await loop.run_in_executor(
                None, functools.partial(self._process_image_to_base64, img_tensor)
            )
            content_payload.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}}
            )

        content_payload.append({"type": "text", "text": prompt_text})
        return content_payload, state

    async def _generate_actions_async(self, batch: dict[str, Tensor]) -> Tensor:
        device = self.dummy_param.device
        batch_size = batch[OBS_STATE].shape[0]

        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        if SERVER_PROFILE:
            requests.post(f"{self.config.vllm_url}/start_profile", timeout=1)

        async with AsyncOpenAI(
            base_url=self.config.vllm_url + "v1",
            api_key=self.config.vllm_api_key,
            max_retries=0,
            timeout=30.0,
        ) as client:
            sem = asyncio.Semaphore(64)

            async def process_single_sample_async(i):
                # Build payload (images encoded in background threads)
                content_payload, state = await self._build_prompt_payload(batch, i)

                async with sem:
                    try:
                        response = await client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": content_payload}],
                            max_tokens=self.config.max_decoding_steps,
                            temperature=0.0,
                            extra_body={"structured_outputs": {"grammar": self.grammar_str}},
                        )
                        generated_text = response.choices[0].message.content
                    except Exception as e:
                        logging.error(f"Async vLLM Error sample {i}: {e}")
                        raise RuntimeError("vLLM Request Failed") from e

                    n_expected = self.action_horizon * self.action_dim
                    actions = generated_text.strip().split()
                    valid_actions = [int(a) for a in actions if a.isdigit()]

                    if len(valid_actions) != n_expected:
                        logging.error(actions)
                        raise RuntimeError(f"Invalid action length: {len(valid_actions)} vs {n_expected}")

                    indices = torch.tensor(valid_actions, device=device).clamp(
                        0, self.config.n_action_bins - 1
                    )
                    action_tensor = bin_centers[indices].view(self.action_horizon, self.action_dim)

                    if self.config.relative_actions:
                        action_tensor = action_tensor + state.unsqueeze(0)

                    return action_tensor

            tasks = [process_single_sample_async(i) for i in range(batch_size)]
            results = await asyncio.gather(*tasks)

        if SERVER_PROFILE:
            requests.post(f"{self.config.vllm_url}/stop_profile", timeout=10)
            exit(0)

        return torch.stack(results)

    async def stream_actions_async(
        self, batch: dict[str, Tensor]
    ) -> AsyncGenerator[tuple[int, Tensor], None]:
        """
        Streams actions one by one as they are generated by the VLM.
        """
        device = self.dummy_param.device
        batch_size = batch[OBS_STATE].shape[0]
        if batch_size > 1:
            raise NotImplementedError("Streaming supported for batch_size=1 only.")

        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        content_payload, state = await self._build_prompt_payload(batch, 0)

        async with AsyncOpenAI(
            base_url=self.config.vllm_url + "v1",
            api_key=self.config.vllm_api_key,
        ) as client:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_payload}],
                max_tokens=self.config.max_decoding_steps,
                temperature=0.0,
                extra_body={"structured_outputs": {"grammar": self.grammar_str}},
                stream=True,
            )

            current_buffer = ""
            found_indices = []
            action_idx = 0

            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if not token:
                    continue

                current_buffer += token

                if " " in current_buffer:
                    parts = current_buffer.split()
                    # The last part might be an incomplete number (e.g. "12" of "123")
                    # unless the token ended with a space.
                    complete_numbers = parts[:-1] if not current_buffer.endswith(" ") else parts
                    current_buffer = parts[-1] if not current_buffer.endswith(" ") else ""

                    for num_str in complete_numbers:
                        if num_str.isdigit():
                            found_indices.append(int(num_str))

                        # Once we have enough indices for one full action
                        if len(found_indices) == self.action_dim:
                            idx_tensor = torch.tensor(found_indices, device=device)
                            action = bin_centers[idx_tensor]
                            if self.config.relative_actions:
                                action = action + state

                            yield (action_idx, action.unsqueeze(0))
                            action_idx += 1
                            found_indices = []

            # Handle any remaining buffer after stream ends because
            # the last token may not end with space
            complete_numbers = current_buffer.split()

            for num_str in complete_numbers:
                if num_str.isdigit():
                    found_indices.append(int(num_str))

            if len(found_indices) == self.action_dim:
                idx_tensor = torch.tensor(found_indices, device=device)
                action = bin_centers[idx_tensor]
                if self.config.relative_actions:
                    action = action + state

                yield (action_idx, action.unsqueeze(0))
            else:
                logging.error(f"Incomplete action at end of stream: {found_indices}")

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return asyncio.run(self._generate_actions_async(batch))
