#!/usr/bin/env bash
# Q4_K_M redo: pass 1 ran against a file that never downloaded (hf --include dropped the first pattern).
# Chained after pipe_5090_q3_quality.sh. Quality + speed + llama-bench + perplexity, then restore AWQ.
set -uo pipefail; source "$(dirname "$0")/common.sh"
until grep -q Q3Q_DONE $C/logs/pipe_5090_q3_quality.log && [ -f $G/Q4_OK ]; do sleep 20; done
mark Q4R_START
pkill -f "vllm serve $MODELS/MiniCPM-V-4.7"; for i in $(seq 1 30); do pgrep -f "vllm serve $MODELS/MiniCPM-V-4.7" >/dev/null || break; sleep 2; done; sleep 5
QB=http://127.0.0.1:18112/v1; QM=MiniCPM-V-4.7-35B-A3B-q4; R=$C/results/q4; mkdir -p $R
LOG_TAG=_redo bash $C/scripts/serve_llama_gguf_5090.sh q4 18112
wait_ready $QB $C/logs/llama_q4_18112.pid && mark Q4_READY || { mark Q4R_SERVER_FAILED; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv > $R/vram_5090_loaded.txt
six_off $QB $QM $R/sixcat_strict_off; mark "Q4_SIXCAT_EXIT $?"
vlm $QB $QM $R/vlm_16x 16; mark "Q4_VLM_EXIT $?"
$VPY $C/scripts/needle_ladder.py --base-url $QB --model $QM --out $R/needle.json > $R/needle.stdout 2>&1; mark "Q4_NEEDLE_EXIT $?"
$VPY $C/scripts/decode_sweep.py --base-url $QB --model $QM --out $R/decode_sweep_5090.json > $R/decode_sweep.stdout 2>&1; mark "Q4_SWEEP_EXIT $?"
kill "$(cat $C/logs/llama_q4_18112.pid)"; sleep 8
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
$LLB/llama-bench -m $G/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -ngl 999 -fa 1 -p 512,4096 -n 128 -r 3 -o json > $C/results/llama_bench_Q4_K_M.json 2> $C/results/llama_bench_Q4_K_M.stderr; mark "BENCH_Q4_EXIT $?"
$LLB/llama-perplexity -m $G/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf -f $C/fixtures/wikitext2_test.raw -ngl 999 -fa on -c 2048 --chunks 60 > $C/results/ppl_Q4_K_M.txt 2>&1; mark "PPL_Q4_EXIT $?"
LOG_TAG=_restored3 bash $C/scripts/serve_vllm_awq_5090.sh; wait_ready http://127.0.0.1:18111/v1 $C/logs/vllm_awq_5090_18111.pid && mark AWQ_RESTORED || mark AWQ_RESTORE_FAILED
mark Q4R_DONE
