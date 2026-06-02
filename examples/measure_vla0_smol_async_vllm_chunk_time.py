#!/usr/bin/env python

from __future__ import annotations

import asyncio
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_times(name: str, times_ms: list[float]) -> None:
    if not times_ms:
        print(f"{name}_count: 0")
        return

    times = torch.tensor(times_ms, dtype=torch.float64)
    print(f"{name}_count: {times.numel()}")
    print(f"{name}_mean_ms: {times.mean().item():.2f}")
    print(f"{name}_p50_ms: {times.median().item():.2f}")
    print(f"{name}_min_ms: {times.min().item():.2f}")
    print(f"{name}_max_ms: {times.max().item():.2f}")


async def generate_streamed_action_chunk(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    actions = []
    async for _, action in model.stream_actions_async(batch):
        actions.append(action.unsqueeze(1))

    if not actions:
        raise RuntimeError("Async vLLM stream did not yield any actions.")

    return torch.cat(actions, dim=1)


config_path = Path("configs/eval_vla0_smol_async_vllm_streaming_libero.json")

print(f"Loading config: {config_path}")
with open(config_path) as f:
    cfg = json.load(f)
print("Config loaded.")

policy_cfg_dict = dict(cfg["policy"])
policy_type = policy_cfg_dict.pop("type")
policy_cfg = make_policy_config(policy_type, **policy_cfg_dict)
print("Policy config created.")

eval_cfg = cfg.get("eval", {})
batch_size = 1
n_batches = 50
warmup_batches = 5

device = torch.device(policy_cfg.device)

dataset_cfg = {"repo_id": "HuggingFaceVLA/smol-libero"}
source = dataset_cfg["repo_id"]
print(f"Loading dataset metadata: {source}")
dataset_metadata = LeRobotDatasetMetadata(
    source,
    root=dataset_cfg.get("root"),
    revision=dataset_cfg.get("revision"),
)
print("Dataset metadata loaded.")

features = dataset_to_policy_features(dataset_metadata.features)
policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
policy_cfg.input_features = {key: ft for key, ft in features.items() if key not in policy_cfg.output_features}

delta_timestamps = {
    "action": [i / dataset_metadata.fps for i in policy_cfg.action_delta_indices],
}
dataset = LeRobotDataset(
    source,
    root=dataset_cfg.get("root"),
    episodes=dataset_cfg.get("episodes"),
    delta_timestamps=delta_timestamps,
    revision=dataset_cfg.get("revision"),
)
print("Dataset created.")
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
dl_iter = cycle(dataloader)
print("Dataloader created.")

print("Creating policy.")
policy = make_policy(cfg=policy_cfg, ds_meta=dataset_metadata)
print("Policy created.")

print("Creating preprocessor.")
preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
print("Preprocessor created.")


def get_batch():
    raw_batch = next(dl_iter)
    return preprocessor(raw_batch)


policy.eval()

autocast = torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext()
times_ms = []
last_actions = None


async def run_measurement() -> None:
    global last_actions

    with torch.inference_mode(), autocast:
        print(f"Running warmup batches: {warmup_batches}")
        for _ in range(warmup_batches):
            batch = get_batch()
            policy.reset()
            _ = await generate_streamed_action_chunk(policy.model, batch)
        print("Warmup done.")

        policy.model.reset_generate_one_action_timing()

        print(f"Measuring batches: {n_batches}")
        for _ in tqdm(range(n_batches), desc="Measuring chunks"):
            batch = get_batch()

            policy.reset()
            sync(device)
            start = time.perf_counter()
            last_actions = await generate_streamed_action_chunk(policy.model, batch)
            sync(device)

            times_ms.append((time.perf_counter() - start) * 1000.0)
        print("Measurement done.")


try:
    asyncio.run(run_measurement())
finally:
    if hasattr(policy.model, "shutdown"):
        policy.model.shutdown()

times = torch.tensor(times_ms)
print(f"config: {config_path}")
print(f"source: {source}")
print(f"batch_size: {batch_size}")
print(f"n_batches: {n_batches}")
print(f"action_chunk_shape: {tuple(last_actions.shape)}")
print(f"mean_ms: {times.mean().item():.2f}")
print(f"p50_ms: {times.median().item():.2f}")
print(f"min_ms: {times.min().item():.2f}")
print(f"max_ms: {times.max().item():.2f}")

summarize_times(
    "generate_one_action_new_obs_true",
    policy.model.get_generate_one_action_new_obs_true_timings_ms(),
)
summarize_times(
    "generate_one_action_new_obs_false",
    policy.model.get_generate_one_action_new_obs_false_timings_ms(),
)
