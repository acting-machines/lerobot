vllm serve /home/olegbalakhnov/vla-0-smol \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --speculative_config '{"model": "/home/olegbalakhnov/lerobot/pusht_mtp_eagle", "num_speculative_tokens": 5, "method": "eagle3"}' \
  --profiler-config '{"torch_profiler_dir": "./vllm_profile"}'
