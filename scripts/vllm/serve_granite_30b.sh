#!/usr/bin/env bash
set -euo pipefail
# Granite-4.1-30B-FP8 (paper leaderboard)
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_NAME:-ibm-granite/granite-4.1-30b-fp8}"
PORT="${VLLM_PORT:-8030}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --trust-remote-code \
  --tool-call-parser granite4 \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
