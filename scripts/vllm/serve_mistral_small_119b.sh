#!/usr/bin/env bash
set -euo pipefail
# Mistral-Small-4-119B (paper leaderboard)
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_NAME:-mistralai/Mistral-Small-4-119B-2603}"
PORT="${VLLM_PORT:-8119}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --tool-call-parser mistral \
  --reasoning-parser mistral \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
