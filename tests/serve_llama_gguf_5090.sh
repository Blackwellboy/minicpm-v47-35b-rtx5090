#!/usr/bin/env bash
# Serve an official MiniCPM-V 4.7 GGUF on the RTX 5090 with llama.cpp (build >= PR #29416).
# usage: serve_llama_gguf_5090.sh <q4|q3> [port]
set -euo pipefail
ARM="${1:?q4|q3}"; PORT="${2:-18112}"
G=$MODELS/MiniCPM-V-4.7-35B-A3B-gguf
case "$ARM" in
  q4) GGUF=$G/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ;;
  q3) GGUF=$G/MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf ;;
  *) exit 2 ;;
esac
BIN="${LLAMA_BIN:-$SRC/llama.cpp-minicpmv47-20261013/build/bin/llama-server}"
CTX="${CTX:-131072}"; NP="${NP:-16}"   # unified KV: 16 slots share 131072 tokens, so one request can still go long
CAMP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"; LOGDIR="$CAMP_ROOT/logs"; mkdir -p "$LOGDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-0}"   # 0 = RTX 5090, 1 = RTX 3090
# Sampling defaults follow the CookBook except repeat-penalty (CookBook 1.05); 1.0 keeps the quant comparison matched to vLLM.
nohup "$BIN" -m "$GGUF" --mmproj "$G/mmproj-model-f16.gguf" \
  --alias "MiniCPM-V-4.7-35B-A3B-$ARM" --host 127.0.0.1 --port "$PORT" \
  -ngl 999 -fa on -c "$CTX" --parallel "$NP" --kv-unified --jinja --reasoning-format deepseek \
  --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty "${REPEAT_PENALTY:-1.0}" \
  --no-webui --metrics ${EXTRA_ARGS:-} > "$LOGDIR/llama_${ARM}_${PORT}${LOG_TAG:-}.log" 2>&1 &
echo $! > "$LOGDIR/llama_${ARM}_${PORT}.pid"; echo "pid=$(cat "$LOGDIR/llama_${ARM}_${PORT}.pid")"
