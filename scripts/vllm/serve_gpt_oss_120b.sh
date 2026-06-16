#!/usr/bin/env bash
set -euo pipefail
# GPT-OSS-120B (paper leaderboard)
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_NAME:-openai/gpt-oss-120b}"
PORT="${VLLM_PORT:-8120}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
