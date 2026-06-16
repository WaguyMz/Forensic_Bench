#!/usr/bin/env bash
set -euo pipefail
# Llama-3.3-70B-Instruct-FP8 (paper leaderboard)
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_NAME:-nvidia/Llama-3.3-70B-Instruct-FP8}"
PORT="${VLLM_PORT:-8070}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --trust-remote-code \
  --tool-call-parser llama3_json \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
