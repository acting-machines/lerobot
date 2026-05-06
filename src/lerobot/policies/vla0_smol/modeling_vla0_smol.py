#!/usr/bin/env python

import asyncio
import contextlib
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.temporal_ensembler import VLA0TemporalEnsembler
from lerobot.policies.vla0_smol.vla0_smol_local import VLA0Local

try:
    from lerobot.policies.vla0_smol.vla0_smol_remote import VLA0Client

    HAS_REMOTE_DEPS = True
except ImportError:
    HAS_REMOTE_DEPS = False


SLEEP_INTERVAL = 0.005


class VLA0SmolPolicy(PreTrainedPolicy):
    """Wrapper class around VLA0 model to train and run inference within LeRobot."""

    config_class = VLA0SmolConfig
    name = "vla0_smol"

    def __init__(
        self,
        config: VLA0SmolConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        if self.config.use_remote_client:
            logging.info("VLA0 Policy: Initializing in REMOTE CLIENT mode (vLLM).")
            if not HAS_REMOTE_DEPS:
                raise ImportError("Please install `openai` and `pillow` for remote inference.")
            self.model = VLA0Client(config)
            if self.config.use_streaming:
                self.service = AsyncInferenceService(self.model)
        else:
            logging.info("VLA0 Policy: Initializing in LOCAL TRAINING mode (PyTorch).")
            self.model = VLA0Local(config)

        self.use_ensembling = self.config.ensemble_size > 1
        if self.use_ensembling:
            self.temporal_ensembler = VLA0TemporalEnsembler(
                ensemble_prediction_count=self.config.ensemble_size
            )
            logging.info("Ensemble mode for token prediction is enabled.")
            assert config.n_action_steps == 0, (
                "When ensemble mode is enabled, n_action_steps param should be zero."
            )
        else:
            self.temporal_ensembler = None
            logging.info("N actions step mode for token prediction is enabled.")
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.should_run_model = True

        self._stream_step_counter = 0

        if hasattr(self.model, "reset"):
            self.model.reset()

        if self.use_ensembling:
            self.temporal_ensembler.reset()

        if self.config.use_remote_client and self.config.use_streaming:
            self.service.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for VLA0")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.use_remote_client and self.config.use_streaming:
            return self.select_action_remote_streaming(batch)
        else:
            return self.select_action_common(batch)

    def select_action_common(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.use_ensembling:
            actions = self.model.generate_actions(batch)

            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            return self.temporal_ensembler.update(actions)
        elif self.config.use_streaming:
            next_action = self.model.generate_one_action(batch).squeeze(1)
            return next_action
        else:
            # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
            # querying the policy.
            if len(self._action_queue) == 0:
                actions = self.model.generate_actions(batch)
                actions = actions[:, : self.config.n_action_steps]

                original_action_dim = self.config.action_feature.shape[0]
                actions = actions[:, :, :original_action_dim]
                # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
                # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
                self._action_queue.extend(actions.transpose(0, 1))
            return self._action_queue.popleft()

    def select_action_remote_streaming(self, batch: dict[str, Tensor]) -> Tensor:
        if self.use_ensembling:
            raise NotImplementedError("Ensemble mode not implemented for vLLM client yet.")

        if self.should_run_model:
            self.service.submit_request(
                batch,
                self._streaming_callback,
            )
            self.should_run_model = False
            self._stream_step_counter = 0

        while not self._action_queue:
            time.sleep(SLEEP_INTERVAL)

        action = self._action_queue.popleft()

        self._stream_step_counter += 1
        if self._stream_step_counter >= self.config.n_action_steps:
            self.should_run_model = True

        return action

    def _streaming_callback(self, stream_item):
        if isinstance(stream_item, (tuple, list)) and len(stream_item) == 2:
            idx, action = stream_item
        else:
            idx, action = -1, stream_item

        # Only append if valid tensor
        if isinstance(action, Tensor) and (idx == -1 or idx < self.config.n_action_steps):
            self._action_queue.append(action)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = self.model.forward(batch)
        loss = loss_dict.pop("loss")
        return loss, loss_dict

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

        # safe MTP model
        if hasattr(self.model, "mtp_model"):
            mtp_model = self.model.mtp_model
            save_model_as_safetensor(mtp_model, str(save_directory / "mtp_model.safetensors"))


class AsyncInferenceService:
    def __init__(self, model_client):
        self.model = model_client
        # Queue stores: (batch, callback_function)
        self.request_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def reset(self) -> None:
        try:
            while not self.request_queue.empty():
                self.request_queue.get_nowait()
        except queue.Empty:
            pass
        self._stop_event.clear()

    def submit_request(self, batch, callback) -> None:
        # Put the last request into the queue, discarding any previous one
        if self.request_queue.full():
            with contextlib.suppress(queue.Empty):
                self.request_queue.get_nowait()
        self.request_queue.put_nowait((batch, callback))

    def _worker_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._stop_event.is_set():
            try:
                batch, callback = self.request_queue.get(timeout=SLEEP_INTERVAL)
                loop.run_until_complete(self._stream_task(batch, callback))
            except queue.Empty:
                continue

    async def _stream_task(self, batch, callback):
        try:
            async for action in self.model.stream_actions_async(batch):
                callback(action)

        except Exception as e:
            logging.error(f"Streaming error: {e}")
            raise RuntimeError("vLLM Streaming Failed") from e
