import copy
import random
import time

import torch
import xgrammar as xgr
from torch import Tensor, nn
from torch.profiler import record_function
from torchvision.transforms import CenterCrop, RandomCrop
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.smolvlm.image_processing_smolvlm_fast import SmolVLMImageProcessorFast

from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.monkey_patch import patch_SmolVLM_amp, patch_SmolVLMProcessor
from lerobot.policies.vla0_smol.mtp_model import MTPModel
from lerobot.policies.vla0_smol.vla0_smol_common import EPS, build_exact_n_numbers_grammar
from lerobot.utils.constants import ACTION, OBS_STATE

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class VLA0Local(nn.Module):
    def __init__(self, config: VLA0SmolConfig):
        super().__init__()
        self.config = config

        self.precision = PRECISION.get(config.precision, torch.float32)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.config.vlm_checkpoint, dtype=self.precision
        )

        # Patch SmolVLMProcessor to enable using SmolVLMImageProcessorFast
        patch_SmolVLMProcessor()

        # Patch SmolVLM to enable AMP training
        patch_SmolVLM_amp(False)

        image_processor = SmolVLMImageProcessorFast.from_pretrained(
            self.config.vlm_checkpoint,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.vlm_checkpoint,
            image_processor=image_processor,
            use_fast=True,
        )

        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]

        if config.freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        self.image_keys = self.config.image_features.keys()

        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.random_crop_fn = RandomCrop(config.crop_shape)
            self.center_crop_fn = CenterCrop(config.crop_shape)

        self.actions_mask_symbol = "<MASK_ACT>"
        assert self.actions_mask_symbol not in self.processor.tokenizer.get_vocab(), (
            f"Replace {self.actions_mask_symbol} token with a different token."
        )
        self.processor.tokenizer.add_tokens([self.actions_mask_symbol], special_tokens=True)
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
        self.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.actions_mask_symbol)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.processor.tokenizer)
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        total_actions = self.config.chunk_size * self.config.action_feature.shape[0]
        ebnf_string = build_exact_n_numbers_grammar(total_actions, 0, self.config.n_action_bins)
        self.compiled_grammar = self.grammar_compiler.compile_grammar(ebnf_string)

        # stream generation
        self.new_obs = True
        self.prefill_times_ms: list[float] = []
        self.generate_one_action_new_obs_true_times_ms: list[float] = []
        self.generate_one_action_new_obs_false_times_ms: list[float] = []

        # multi-token prediction
        self.train_mtp = config.num_train_mtp_heads > 0
        self.inference_mtp = config.num_inference_mtp_heads > 0

        if self.train_mtp:
            self.mtp_model = MTPModel(
                num_heads=self.config.num_train_mtp_heads,
                config=self.vlm.model.text_model.config,
                input_embedding=self.vlm.get_input_embeddings(),
                output_embedding=self.vlm.get_output_embeddings(),
            )

    def apply_action_masking(self, actions: list[list[str]]):
        if not self.training:
            return actions

        if random.random() < self.config.action_mask_skip_per:
            return actions

        num_actions = len(actions)

        aug_per = random.uniform(0.0, self.config.action_mask_aug_per)
        num_actions_to_mask = int(num_actions * aug_per)

        if num_actions_to_mask > 0:
            indices = random.sample(range(num_actions), num_actions_to_mask)

            for idx in indices:
                actions[idx] = self.actions_mask_symbol

        return actions

    def create_prefix_tokens(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        prefix_text: list[str],
        actions: torch.Tensor | None,
    ):
        device = states.device
        batch_size = states.shape[0]

        # Precompute bin edges on GPU
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)[:-1]

        if actions is None:
            disc_actions_cpu = [""] * batch_size
        else:
            if self.config.relative_actions:
                actions = actions - states.unsqueeze(1)
            discretized_actions = torch.bucketize(actions, bins) - 1  # shape: [B, state_dim]
            disc_actions_cpu = discretized_actions.detach().cpu().numpy()

        # Build strings in batch
        prompts = []
        for prefix, act in zip(prefix_text, disc_actions_cpu, strict=False):
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in range(len(images))],
                        {
                            "type": "text",
                            "text": prefix,
                        },
                    ],
                }
            ]

            if actions is not None:
                action_list = list(map(str, act.flatten().tolist()))
                action_list = self.apply_action_masking(action_list)
                action_str = " ".join(action_list)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{action_str}",
                            },
                        ],
                    }
                )
            prompts.append(
                self.processor.apply_chat_template(messages, add_generation_prompt=actions is None)
            )

        images = {
            camera_name: list(torch.unbind(camera_images, dim=0))
            for camera_name, camera_images in images.items()
        }

        images_reshaped = []
        for imgs in zip(*images.values(), strict=True):
            if self.do_crop:
                crop_fn = self.random_crop_fn if self.training else self.center_crop_fn
                images_reshaped.append([crop_fn(img) for img in imgs])
            else:
                images_reshaped.append(list(imgs))

        prefix_out = self.processor(
            images=images_reshaped,
            text=prompts,
            do_resize=self.config.do_image_splitting,
            do_rescale=False,
            return_tensors="pt",
            padding=True,
            padding_side="right" if actions is not None else "left",
        )
        return prefix_out

    def create_input_tokens(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        prefix_text: list[str],
        actions: torch.Tensor | None = None,
    ):
        device = states.device

        prefix_out = self.create_prefix_tokens(
            states=states, images=images, prefix_text=prefix_text, actions=actions
        )
        prefix_out = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in prefix_out.items()}

        if actions is None:
            loss_mask = None
        else:
            split_mask = torch.where(prefix_out["input_ids"] == self.config.start_actions_token, 1, 0)
            loss_mask = torch.cumsum(split_mask, dim=-1).clamp(0, 1) & prefix_out["attention_mask"]
            is_masked_token = prefix_out["input_ids"] == self.mask_token_id
            loss_mask = loss_mask & (~is_masked_token)

        return prefix_out, loss_mask

    def prepare_images(self, batch: torch.Tensor):
        """Preprocess LeRobot batch into inputs"""
        images = {}
        present_img_keys = [key for key in self.image_keys if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in self.image_keys:
            if key in present_img_keys:
                img = batch[key]

            images[key] = img
        return images

    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        with record_function("create_input_tokens"):
            images = self.prepare_images(batch)

            padded_outs, loss_mask = self.create_input_tokens(
                states=batch[OBS_STATE],
                images=images,
                prefix_text=batch["prefix"],
                actions=batch[ACTION],
            )

        with record_function("forward"):
            outputs = self.vlm.forward(
                input_ids=padded_outs["input_ids"],
                attention_mask=padded_outs["attention_mask"],
                pixel_values=padded_outs["pixel_values"],
                pixel_attention_mask=padded_outs["pixel_attention_mask"],
                use_cache=self.config.use_cache,
                output_hidden_states=True,
                return_dict=True,
            )

        with record_function("loss"):
            logits = outputs.logits
            logits = logits.to(torch.float32)

            loss_fct = nn.CrossEntropyLoss(reduction="none")

            # Shift left for next-step prediction
            logits = logits[:, :-1, :]
            targets = padded_outs["input_ids"][:, 1:].to(device)  # Shift targets
            loss_mask_vlm = loss_mask[:, 1:].to(device)  # Ensure correct shape

            # Compute per-token loss
            token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            # Apply loss mask
            token_loss = token_loss * loss_mask_vlm.reshape(-1)

            # Compute final loss
            vlm_loss = token_loss.sum() / torch.clamp(loss_mask_vlm.sum(), min=1)

        if not self.train_mtp:
            loss_dict = {
                "vlm_loss": vlm_loss.item(),
                "loss": vlm_loss,
                "sequence_len": padded_outs["input_ids"].shape[-1],
            }
            return loss_dict

        with record_function("mtp_loss"):
            base_hidden_states = [outputs.hidden_states[id] for id in self.config.mtp_layers_ids]
            fused_hidden_state = self.mtp_model.fuse_base_model_hidden_states(base_hidden_states)[:, :-1, :]

            mtp_losses = self.mtp_model.calculate_mtp_loss(
                input_ids=padded_outs["input_ids"][:, 1:],
                hidden_states=fused_hidden_state,
                loss_mask=loss_mask[:, 1:],
            )
            mtp_loss = sum(mtp_losses)

            loss = vlm_loss + mtp_loss

            loss_dict = {
                "vlm_loss": vlm_loss.item(),
                "mtp_loss": mtp_loss.item(),
                "loss": loss,
                "sequence_len": padded_outs["input_ids"].shape[-1],
            }

        return loss_dict

    def generate_next_token(self, input_ids, past_key_values):
        with torch.inference_mode():
            out = self.vlm(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
        generated_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return generated_token, out

    def check_end_of_generation(self, generated_token=None):
        if generated_token is not None:
            for i, token in enumerate(generated_token):
                if self.generation_finished[i]:
                    continue
                elif token == self.eos_token_id or token == self.pad_token_id:
                    self.generation_finished[i] = True

        return sum(self.generation_finished) == len(self.generation_finished)

    def reconstruct_actions(self, decoded_actions, batch):
        batch_size = batch[OBS_STATE].shape[0]
        device = batch[OBS_STATE].device
        # print(f"decoded actions: {decoded_actions}")
        discretized_actions = torch.stack(decoded_actions, dim=0).reshape(batch_size, -1, self.action_dim)

        # Assuming same bin setup
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)

        # Compute bin centers (midpoints between edges)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # shape: [n_state_bins]

        # Map discretized indices back to continuous states
        reconstructed_actions = bin_centers[discretized_actions.clamp(0, self.config.n_state_bins - 1)]
        if self.config.relative_actions:
            reconstructed_actions += batch[OBS_STATE].unsqueeze(1)

        return reconstructed_actions

    def reset_prefill_timing(self):
        self.prefill_times_ms.clear()

    def get_prefill_timings_ms(self) -> list[float]:
        return list(self.prefill_times_ms)

    def reset_generate_one_action_timing(self):
        self.generate_one_action_new_obs_true_times_ms.clear()
        self.generate_one_action_new_obs_false_times_ms.clear()

    def get_generate_one_action_new_obs_true_timings_ms(self) -> list[float]:
        return list(self.generate_one_action_new_obs_true_times_ms)

    def get_generate_one_action_new_obs_false_timings_ms(self) -> list[float]:
        return list(self.generate_one_action_new_obs_false_times_ms)

    @staticmethod
    def _synchronize_timing_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _record_generate_one_action_timing(
        self, device: torch.device, start_time: float, started_with_new_obs: bool
    ) -> None:
        self._synchronize_timing_device(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if started_with_new_obs:
            self.generate_one_action_new_obs_true_times_ms.append(elapsed_ms)
        else:
            self.generate_one_action_new_obs_false_times_ms.append(elapsed_ms)

    def prefill(self, batch):
        timing_device = batch[OBS_STATE].device
        self._synchronize_timing_device(timing_device)
        start_time = time.perf_counter()

        images = self.prepare_images(batch)
        batch_size = batch[OBS_STATE].shape[0]

        padded_outs, _ = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=images,
            prefix_text=batch["prefix"],
            actions=None,
        )

        self.prefix_len = padded_outs["input_ids"].shape[1] - 1
        self.input_ids = torch.full(
            (batch_size, self.prefix_len + self.config.max_decoding_steps),
            self.pad_token_id,
            device=padded_outs["input_ids"].device,
            dtype=padded_outs["input_ids"].dtype,
        )
        self.input_ids[:, : self.prefix_len] = padded_outs["input_ids"][:, 1:]

        with torch.inference_mode():
            out = self.vlm(**padded_outs, use_cache=True, output_hidden_states=True)

        generated_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        self._synchronize_timing_device(generated_token.device)
        self.prefill_times_ms.append((time.perf_counter() - start_time) * 1000.0)
        return generated_token, out

    def initialise_new_generation(self, batch):
        batch_size = batch[OBS_STATE].shape[0]

        self.new_obs = False
        self.generation_batch = batch
        self.generation_finished = [False] * batch_size
        self.action_index = 0
        self.input_idx_base = 0
        self.input_idx_mtp = 0
        self.input_ids_len = 0

    def generate_one_action(self, batch):
        device = batch[OBS_STATE].device
        batch_size = batch[OBS_STATE].shape[0]
        started_with_new_obs = self.new_obs
        self._synchronize_timing_device(device)
        start_time = time.perf_counter()

        next_action_is_generated = [False] * batch_size
        decoded_actions = [None] * batch_size

        if self.new_obs:
            self.initialise_new_generation(batch)

            generated_token, output = self.prefill(batch=batch)

            self.input_ids_len = self.prefix_len
            self.input_idx_base = self.prefix_len
            self.input_idx_mtp = 0

            self.input_ids[:, self.input_ids_len : self.input_ids_len + 1] = generated_token
            self.input_ids_len += 1

            self.past_key_values = output.past_key_values

            if self.inference_mtp:
                self.mtp_past_key_values = DynamicCache(config=self.mtp_model.cfg)
                base_hidden_states = [output.hidden_states[id] for id in self.config.mtp_layers_ids]
                self.hidden_state = self.mtp_model.fuse_base_model_hidden_states(base_hidden_states)

        # generate one action
        mtp_heads = self.config.num_inference_mtp_heads if self.inference_mtp else 0
        max_remained_steps = int(
            (self.config.max_decoding_steps - (self.input_ids_len - self.prefix_len)) / (mtp_heads + 1)
        )
        for _ in range(max_remained_steps):
            for head_id in range(self.config.num_inference_mtp_heads):
                if head_id == 0:
                    generated_token, out = self.mtp_model.generate_next_token(
                        input_ids=self.input_ids[:, self.input_idx_mtp : self.input_ids_len],
                        hidden_states=self.hidden_state,
                        past_key_values=self.mtp_past_key_values,
                    )
                    local_mtp_past_key_values = Cache(
                        layers=[copy.copy(layer) for layer in self.mtp_past_key_values.layers]
                    )
                    self.input_idx_mtp = self.input_ids_len
                else:
                    generated_token, out = self.mtp_model.generate_next_token(
                        input_ids=self.input_ids[:, self.input_ids_len - 1 : self.input_ids_len],
                        hidden_states=out.last_hidden_state[:, -1:, :],
                        past_key_values=local_mtp_past_key_values,
                    )

                self.input_ids[:, self.input_ids_len : self.input_ids_len + 1] = generated_token
                self.input_ids_len += 1
                self.check_end_of_generation(generated_token)

            generated_token, out = self.generate_next_token(
                input_ids=self.input_ids[:, self.input_idx_base : self.input_ids_len],
                past_key_values=self.past_key_values,
            )
            self.input_idx_base = self.input_ids_len
            self.input_ids[:, self.input_ids_len : self.input_ids_len + 1] = generated_token
            self.input_ids_len += 1

            self.check_end_of_generation(generated_token)

            if self.inference_mtp:
                base_hidden_states = [out.hidden_states[id] for id in self.config.mtp_layers_ids]
                self.hidden_state = self.mtp_model.fuse_base_model_hidden_states(base_hidden_states)

            # decode every new sequence and count amount of spaces
            decoded_texts = self.processor.batch_decode(
                self.input_ids[:, self.prefix_len : self.input_ids_len],
                skip_special_tokens=True,
            )  # return list of lists
            for i in range(batch_size):
                if next_action_is_generated[i]:
                    continue

                output = decoded_texts[i].strip().split()

                # if output is invalid just add zeros
                if not all(a.isdigit() and 0 <= int(a) < self.config.n_state_bins for a in output):
                    next_action_is_generated[i] = True
                    decoded_actions[i] = torch.zeros(self.action_dim, device=device, dtype=torch.long)

                # check if we finished next action generation
                if self.generation_finished[i] or len(output) > self.action_dim * (self.action_index + 1):
                    next_action_is_generated[i] = True
                    action_text = output[
                        self.action_dim * (self.action_index) : self.action_dim * (self.action_index + 1)
                    ]
                    decoded_actions[i] = torch.tensor([int(a) for a in action_text], device=device)

            if sum(next_action_is_generated) == len(next_action_is_generated):
                self.action_index += 1
                if self.action_index == self.config.n_action_steps:
                    self.new_obs = True
                result = self.reconstruct_actions(decoded_actions, self.generation_batch)
                self._record_generate_one_action_timing(device, start_time, started_with_new_obs)
                return result

        result = torch.zeros((batch_size, 1, self.action_dim), device=device, dtype=torch.long)
        self._record_generate_one_action_timing(device, start_time, started_with_new_obs)
        return result

    def generate_actions(self, batch: dict[str, torch.Tensor]):
        actions = []
        self.new_obs = True

        for _ in range(self.config.n_action_steps):
            action = self.generate_one_action(batch=batch)
            actions.append(action)

        action_chunk = torch.cat(actions, dim=1)
        return action_chunk
