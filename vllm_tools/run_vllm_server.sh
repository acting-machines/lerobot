vllm serve olegbalakhnov/vla-0-smol-libero-vllm \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --profiler-config '{"torch_profiler_dir": "./vllm_profile"}' \
  --attention-backend TRITON_ATTN \
  --speculative_config '{"model": "olegbalakhnov/vla-0-smol-libero-vllm-spec", "num_speculative_tokens": 3, "method": "eagle3"}' \
