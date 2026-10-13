#!/usr/bin/env bash
# Q3 QUALITY on the 5090, chained after pipe_5090.sh (the 3090 plan failed: llama.cpp cudaMalloc of the 14.5 GB
# weight buffer fails on the 3090 under WSL after the first attempt, despite 23 GB reported free).
set -uo pipefail; source "$(dirname "$0")/common.sh"
until grep -q P5090_DONE $C/logs/pipe_5090.log; do sleep 20; done
mark Q3Q_START
pkill -f "vllm serve $MODELS/MiniCPM-V-4.7"; for i in $(seq 1 30); do pgrep -f "vllm serve $MODELS/MiniCPM-V-4.7" >/dev/null || break; sleep 2; done; sleep 5
B=http://127.0.0.1:18112/v1; M=MiniCPM-V-4.7-35B-A3B-q3; R=$C/results/q3; mkdir -p $R
LOG_TAG=_quality bash $C/scripts/serve_llama_gguf_5090.sh q3 18112
wait_ready $B $C/logs/llama_q3_18112.pid || mark Q3Q_SERVER_FAILED
six_off $B $M $R/sixcat_strict_off; mark "Q3_SIXCAT_EXIT $?"
vlm $B $M $R/vlm_16x 16; mark "Q3_VLM_EXIT $?"
kill "$(cat $C/logs/llama_q3_18112.pid)"; sleep 8
LOG_TAG=_restored2 bash $C/scripts/serve_vllm_awq_5090.sh; wait_ready http://127.0.0.1:18111/v1 $C/logs/vllm_awq_5090_18111.pid && mark AWQ_RESTORED || mark AWQ_RESTORE_FAILED
mark Q3Q_DONE
