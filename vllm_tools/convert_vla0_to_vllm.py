import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VLA0_Converter")


def convert_checkpoint(input_source: str, output_dir: Path, base_model_id: str | None = None):
    """
    Converts a LeRobot VLA0 checkpoint (Local Path or HF Repo) into a vLLM-compatible format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Input: Local Path vs Hugging Face Repo
    input_path = Path(input_source)
    temp_dir = None
    
    if input_path.exists() and input_path.is_dir():
        logger.info(f"Using local input directory: {input_path}")
        source_dir = input_path
    else:
        logger.info(f"Input path not found locally. Treating '{input_source}' as HF Repo ID...")
        temp_dir = TemporaryDirectory()
        source_dir = Path(temp_dir.name)
        # Download only the weights and essential configs from the source repo
        snapshot_download(
            repo_id=input_source,
            local_dir=source_dir,
            allow_patterns=["*.safetensors", "config.json"],
            local_dir_use_symlinks=False,
        )

    # 2. Determine Base Model ID for Architecture Files
    lerobot_config_path = source_dir / "config.json"
    if not base_model_id and lerobot_config_path.exists():
        try:
            with open(lerobot_config_path) as f:
                lerobot_conf = json.load(f)
                base_model_id = lerobot_conf.get("vlm_checkpoint")
                logger.info(f"Detected base model from config: {base_model_id}")
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")

    if not base_model_id:
        base_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        logger.warning(f"Base model ID not found. Defaulting to: {base_model_id}")

    # 3. Download Architecture Files (Everything but weights)
    allow_patterns = [
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "*.model",
    ]

    logger.info(f"Downloading architecture files from {base_model_id}...")
    snapshot_download(
        repo_id=base_model_id,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )

    # 4. Patch Preprocessor
    with open(output_dir / "config.json") as f:
        target_vocab_size = json.load(f).get("vocab_size", 49152)
    logger.info(f"Target Vocab Size (from base config): {target_vocab_size}")

    preprocessor_path = output_dir / "preprocessor_config.json"
    if preprocessor_path.exists():
        with open(preprocessor_path) as f:
            proc_config = json.load(f)

        proc_config["size"] = {"longest_edge": 512}
        proc_config["max_image_size"] = {"longest_edge": 512}

        # proc_config["size"] = {"longest_edge": 620}
        # proc_config["max_image_size"] = {"longest_edge": 620}

        with open(preprocessor_path, "w") as f:
            json.dump(proc_config, f, indent=2)

    # 5. Load and Remap Weights
    # Find all safetensors files (handles single file or sharded)
    weight_files = list(source_dir.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"No .safetensors weights found in {source_dir}")

    new_state_dict = {}
    remapped_count = 0

    logger.info(f"Processing weights from {len(weight_files)} file(s)...")
    for wf in weight_files:
        state_dict = load_file(wf)
        for key, value in state_dict.items():
            new_key = key
            # Standardizing prefix
            if key.startswith("model.vlm."):
                new_key = key.replace("model.vlm.", "")
                remapped_count += 1
            elif key.startswith("vlm."):
                new_key = key.replace("vlm.", "")
                remapped_count += 1

            # Vocab Truncation Logic
            if new_key in ["model.text_model.embed_tokens.weight", "lm_head.weight"]:
                current_vocab_size = value.shape[0]
                if current_vocab_size > target_vocab_size:
                    logger.warning(f"TRUNCATING {new_key}: {current_vocab_size} -> {target_vocab_size}")
                    value = value[:target_vocab_size, :]

            if new_key.startswith("model") or new_key.startswith("lm_head"):
                print(f"Mapping {key} -> {new_key}")
                new_state_dict[new_key] = value

    # 6. Save Final Weights
    output_weights_path = output_dir / "model.safetensors"
    save_file(new_state_dict, output_weights_path)

    # Cleanup temp dir if created
    if temp_dir:
        temp_dir.cleanup()

    logger.info(f"\nConversion Complete! Remapped {remapped_count} keys.")
    logger.info(f"vLLM Model Ready at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot VLA0 checkpoint to vLLM format.")
    parser.add_argument("--input", type=str, required=True, help="Local path OR Hugging Face repo ID (e.g. 'lerobot/vla0-smolvlm')")
    parser.add_argument("--output", type=str, required=True, help="Path where to save the vLLM-ready model")
    parser.add_argument("--base", type=str, default=None, help="Force a specific base model ID for architecture files")

    args = parser.parse_args()
    convert_checkpoint(args.input, args.output, args.base)
