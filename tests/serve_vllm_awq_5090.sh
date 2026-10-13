#!/usr/bin/env bash
# Serve MiniCPM-V-4.7-35B-A3B-AWQ (W4A16) on the RTX 5090 (GPU0) with vLLM nightly.
set -euo pipefail
PORT="${PORT:-18111}"
MODEL_DIR="${MODEL_DIR:-$MODELS/MiniCPM-V-4.7-35B-A3B-AWQ-native}"  # see make_native_overlay.py
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"   # SIXCAT thinking-on ceilings reach 32768 output tokens
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SERVED_NAME="${SERVED_NAME:-MiniCPM-V-4.7-35B-A3B-AWQ}"
VLLM_BIN="${VLLM_BIN:-$VENVS/vllm-minicpmv47/bin/vllm}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CAMP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$CAMP_ROOT/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/vllm_awq_5090_${PORT}${LOG_TAG:-}.log"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# WSL2 reports no pin_memory/UVA; force it (same patch as MINICPM5_2B_3090_5090_20260908)
export PYTHONPATH="$CAMP_ROOT/scripts/vllm_wsl_patch${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
"$(dirname "$VLLM_BIN")/python" -c "import torch;n=torch.cuda.get_device_name(0);print('gpu0=',n);assert '5090' in n"
# shellcheck disable=SC2086
nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-batched-tokens 8192 \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --limit-mm-per-prompt '{"image": 5, "video": 1}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  $EXTRA_ARGS >"$LOG" 2>&1 &
echo $! > "$LOGDIR/vllm_awq_5090_${PORT}.pid"
echo "pid=$(cat "$LOGDIR/vllm_awq_5090_${PORT}.pid") log=$LOG"
