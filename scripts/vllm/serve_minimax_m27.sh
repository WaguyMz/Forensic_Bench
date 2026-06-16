#!/usr/bin/env bash
set -euo pipefail
# MiniMax-M2.7 (paper leaderboard)
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_NAME:-MiniMaxAI/MiniMax-M2.7}"
PORT="${VLLM_PORT:-8229}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --trust-remote-code \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
