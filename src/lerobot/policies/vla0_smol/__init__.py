# register the processor steps
from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.modeling_vla0_smol import VLA0SmolPolicy
from lerobot.policies.vla0_smol.processor_vla0_smol import VLA0SmolPrefixProcessorStep

__all__ = ["VLA0SmolPrefixProcessorStep", "VLA0SmolPolicy", "VLA0SmolConfig"]
