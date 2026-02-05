#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.vla0_smol_common import EPS
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="vla0_smol_prefix_processor")
@dataclass
class VLA0SmolPrefixProcessorStep(ComplementaryDataProcessorStep):
    n_state_bins: int
    use_state: bool
    task_key: str = "task"
    prefix_key: str = "prefix"

    def complementary_data(self, complementary_data):
        states = self.transition[TransitionKey.OBSERVATION][OBS_STATE]
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.n_state_bins + 1, device=states.device)[:-1]
        discretized_states = torch.bucketize(states, bins) - 1
        disc_states_cpu = discretized_states.detach().cpu().numpy()

        tasks = complementary_data.get(self.task_key, [""] * states.shape[0])

        prompts = []
        for txt, disc_st in zip(tasks, disc_states_cpu, strict=False):
            task_cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(map(str, disc_st.tolist()))

            if self.use_state:
                prefix = f"Task: {task_cleaned}, State: {state_str}, Actions: "
            else:
                prefix = f"Task: {task_cleaned}, Actions: "
            prompts.append(prefix)

        new_complementary_data = dict(complementary_data)
        new_complementary_data[self.prefix_key] = prompts
        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "n_state_bins": self.n_state_bins,
            "use_state": self.use_state,
            "task_key": self.task_key,
            "prefix_key": self.prefix_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_vla0_smol_pre_post_processors(
    config: VLA0SmolConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the VLA0-Smol policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the VLA0-Smol policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        VLA0SmolPrefixProcessorStep(
            n_state_bins=config.n_state_bins,
            use_state=config.use_state,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
