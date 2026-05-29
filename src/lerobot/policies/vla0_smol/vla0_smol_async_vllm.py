import asyncio
import inspect
import itertools
import logging
import time
from collections.abc import AsyncGenerator

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile
from torchvision.transforms import CenterCrop
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

try:
    from vllm.sampling_params import RequestOutputKind
except ImportError:
    RequestOutputKind = None

try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    StructuredOutputsParams = None

try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.vla0_smol_common import EPS, build_exact_n_numbers_grammar
from lerobot.utils.constants import OBS_STATE

STREAM_FIRST_ACTION_PROFILE = False
STREAM_FIRST_ACTION_TRACE = "trace_vla0_stream_first_action.json"


class VLA0AsyncVLLMClient(nn.Module):
    def __init__(self, config: VLA0SmolConfig):
        super().__init__()
        self.config = config

        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]
        self.image_keys = self.config.image_features.keys()
        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.center_crop_fn = CenterCrop(config.crop_shape)

        self.model_name = self.config.vllm_model

        total_actions = self.config.chunk_size * self.config.action_feature.shape[0]
        self.grammar_str = build_exact_n_numbers_grammar(total_actions, 0, self.config.n_action_bins)

        engine_kwargs = {
            "model": self.model_name,
            "dtype": self.config.precision,
            "max_model_len": self.config.vllm_max_model_len,
            "limit_mm_per_prompt": {"image": max(1, len(self.image_keys))},
            "gpu_memory_utilization": self.config.vllm_gpu_memory_utilization,
            "enforce_eager": self.config.vllm_enforce_eager,
            "mm_processor_kwargs": {
                "do_image_splitting": False,
            },
        }
        if self.config.vllm_attention_backend is not None:
            engine_kwargs["attention_backend"] = self.config.vllm_attention_backend

        self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        self.tokenizer = self.llm.get_tokenizer()
        self._request_counter = itertools.count()

        # Dummy param for device management.
        self.register_buffer("dummy_param", torch.empty(0))
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1)
        self.register_buffer("action_bin_centers", 0.5 * (bins[:-1] + bins[1:]), persistent=False)
        self.delta_sampling_params = self._make_sampling_params(delta_output=True)
        self.delta_output_enabled = self._is_delta_output(self.delta_sampling_params)

        # stream generation timings
        self.generate_one_action_new_obs_true_times_ms: list[float] = []
        self.generate_one_action_new_obs_false_times_ms: list[float] = []

    def forward(self, batch):
        raise NotImplementedError("Async vLLM backend cannot be trained. Use use_async_vllm_client=False.")

    def reset(self) -> None:
        pass

    def shutdown(self) -> None:
        self.llm.shutdown()

    def reset_generate_one_action_timing(self):
        self.generate_one_action_new_obs_true_times_ms.clear()
        self.generate_one_action_new_obs_false_times_ms.clear()

    def get_generate_one_action_new_obs_true_timings_ms(self) -> list[float]:
        return list(self.generate_one_action_new_obs_true_times_ms)

    def get_generate_one_action_new_obs_false_timings_ms(self) -> list[float]:
        return list(self.generate_one_action_new_obs_false_times_ms)

    def _record_stream_action_timing(self, start_time: float, action_idx: int) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - start_time) * 1000.0
        if action_idx == 0:
            self.generate_one_action_new_obs_true_times_ms.append(elapsed_ms)
        else:
            self.generate_one_action_new_obs_false_times_ms.append(elapsed_ms)
        return now

    def _validate_batch_size_one(self, batch: dict[str, Tensor]) -> None:
        batch_size = batch[OBS_STATE].shape[0]
        if batch_size != 1:
            raise NotImplementedError(f"Async vLLM client supports batch_size=1 only, got {batch_size}.")

    def _process_image(self, tensor_img: Tensor) -> Image.Image:
        if self.do_crop:
            tensor_img = self.center_crop_fn(tensor_img)

        # Make contiguous before CPU transfer.
        tensor_img = tensor_img.detach()

        if tensor_img.device.type != "cpu":
            tensor_img = tensor_img.to("cpu", non_blocking=True)

        arr = tensor_img.permute(1, 2, 0).contiguous().numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    def _build_prompt_inputs(self, batch: dict[str, Tensor], index: int) -> tuple[dict, Tensor]:
        state = batch[OBS_STATE][index]
        prompt_text = batch["prefix"][index]

        present_img_keys = [key for key in self.image_keys if key in batch]
        images = []

        for key in present_img_keys:
            img_tensor = batch[key][index]
            images.append(self._process_image(img_tensor))

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        inputs = {"prompt_token_ids": prompt_token_ids}
        if images:
            image_data = images[0] if len(images) == 1 else images
            inputs["multi_modal_data"] = {"image": image_data}
        return inputs, state

    def _make_sampling_params(self, *, delta_output: bool = False) -> SamplingParams:
        kwargs = {
            "temperature": 0.0,
            "max_tokens": self.config.max_decoding_steps,
        }

        signature = inspect.signature(SamplingParams)
        if delta_output and RequestOutputKind is not None and "output_kind" in signature.parameters:
            kwargs["output_kind"] = RequestOutputKind.DELTA

        if "structured_outputs" in signature.parameters and StructuredOutputsParams is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(grammar=self.grammar_str)
        elif "guided_decoding" in signature.parameters:
            guided_decoding_cls = GuidedDecodingParams or StructuredOutputsParams
            if guided_decoding_cls is None:
                logging.warning("vLLM SamplingParams does not expose grammar-guided decoding params.")
            else:
                kwargs["guided_decoding"] = guided_decoding_cls(grammar=self.grammar_str)
        else:
            logging.warning("vLLM SamplingParams does not expose grammar-guided decoding.")

        return SamplingParams(**kwargs)

    def _is_delta_output(self, sampling_params: SamplingParams) -> bool:
        return (
            RequestOutputKind is not None
            and getattr(sampling_params, "output_kind", None) == RequestOutputKind.DELTA
        )

    async def _generate_text_async(self, inputs: dict, sampling_params: SamplingParams) -> str:
        request_id = f"vla0-async-{next(self._request_counter)}"
        generated_text_parts = []
        final_text = ""

        async for output in self.llm.generate(
            prompt=inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            for completion in output.outputs:
                final_text = completion.text
                generated_text_parts.append(completion.text)

            if output.finished:
                break

        if self._is_delta_output(sampling_params):
            return "".join(generated_text_parts)
        return final_text

    def _decode_actions(self, generated_text: str, state: Tensor) -> Tensor:
        device = self.dummy_param.device
        n_expected = self.action_horizon * self.action_dim

        actions = generated_text.strip().split()
        valid_actions = [int(action) for action in actions if action.isdigit()]

        if len(valid_actions) != n_expected:
            logging.error(actions)
            raise RuntimeError(f"Invalid action length: {len(valid_actions)} vs {n_expected}")

        indices = torch.tensor(valid_actions, device=device).clamp(0, self.config.n_state_bins - 1)
        action_tensor = self.action_bin_centers[indices].view(self.action_horizon, self.action_dim)

        if self.config.relative_actions:
            action_tensor = action_tensor + state.unsqueeze(0)

        return action_tensor

    async def _generate_actions_async(self, batch: dict[str, Tensor]) -> Tensor:
        self._validate_batch_size_one(batch)

        inputs, state = self._build_prompt_inputs(batch, 0)
        try:
            generated_text = await self._generate_text_async(inputs, self.delta_sampling_params)
        except Exception as e:
            logging.error(f"Async vLLM error: {e}")
            raise RuntimeError("Async vLLM request failed") from e

        return self._decode_actions(generated_text, state).unsqueeze(0)

    async def stream_actions_async(
        self, batch: dict[str, Tensor]
    ) -> AsyncGenerator[tuple[int, Tensor], None]:
        """
        Streams actions one by one as they are generated by the local async vLLM engine.
        """
        device = self.dummy_param.device
        self._validate_batch_size_one(batch)

        stream_start_time = time.perf_counter()
        previous_action_time = stream_start_time

        inputs, state = self._build_prompt_inputs(batch, 0)
        request_id = f"vla0-stream-{next(self._request_counter)}"
        profiler = None
        if STREAM_FIRST_ACTION_PROFILE:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            profiler = profile(activities=activities, record_shapes=True, with_stack=True)
            profiler.start()

        current_buffer = ""
        found_indices = []
        action_idx = 0
        previous_text = ""

        async for output in self.llm.generate(
            prompt=inputs,
            sampling_params=self.delta_sampling_params,
            request_id=request_id,
        ):
            for completion in output.outputs:
                token = completion.text
                if not self.delta_output_enabled:
                    token = token.removeprefix(previous_text)
                    previous_text = completion.text
                if not token:
                    continue

                current_buffer += token

                if " " in current_buffer:
                    parts = current_buffer.split()
                    complete_numbers = parts[:-1] if not current_buffer.endswith(" ") else parts
                    current_buffer = parts[-1] if not current_buffer.endswith(" ") else ""

                    for num_str in complete_numbers:
                        if num_str.isdigit():
                            found_indices.append(int(num_str))

                        if len(found_indices) == self.action_dim:
                            idx_tensor = torch.tensor(found_indices, device=device).clamp(
                                0, self.config.n_state_bins - 1
                            )
                            action = self.action_bin_centers[idx_tensor]
                            if self.config.relative_actions:
                                action = action + state

                            previous_action_time = self._record_stream_action_timing(
                                previous_action_time if action_idx > 0 else stream_start_time,
                                action_idx,
                            )
                            if action_idx == 0 and profiler is not None:
                                profiler.stop()
                                profiler.export_chrome_trace(STREAM_FIRST_ACTION_TRACE)
                                profiler = None
                            yield (action_idx, action.unsqueeze(0))
                            action_idx += 1
                            found_indices = []

            if output.finished:
                break

        complete_numbers = current_buffer.split()

        for num_str in complete_numbers:
            if num_str.isdigit():
                found_indices.append(int(num_str))

        if len(found_indices) == self.action_dim:
            idx_tensor = torch.tensor(found_indices, device=device).clamp(0, self.config.n_state_bins - 1)
            action = self.action_bin_centers[idx_tensor]
            if self.config.relative_actions:
                action = action + state

            self._record_stream_action_timing(
                previous_action_time if action_idx > 0 else stream_start_time,
                action_idx,
            )
            if action_idx == 0 and profiler is not None:
                profiler.stop()
                profiler.export_chrome_trace(STREAM_FIRST_ACTION_TRACE)
            yield (action_idx, action.unsqueeze(0))
        elif len(found_indices) > 0:
            logging.error(f"Incomplete action at end of stream: {found_indices}")

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return asyncio.run(self._generate_actions_async(batch))
