#!/usr/bin/env bash
# RTX 5090 pipeline: AWQ (as shipped, then card 4x/36-slice settings) -> Q4 full -> Q3 speed -> llama-bench/perplexity -> restore AWQ.
set -uo pipefail; source "$(dirname "$0")/common.sh"
AB=http://127.0.0.1:18111/v1; AM=MiniCPM-V-4.7-35B-A3B-AWQ
stop_vllm() { pkill -f "vllm serve $MODELS/MiniCPM-V-4.7" ; for i in $(seq 1 30); do pgrep -f "vllm serve $MODELS/MiniCPM-V-4.7" >/dev/null || break; sleep 2; done; sleep 5; }
mark P5090_START
# 1. AWQ as shipped (16x, max_slice_nums 9) -- server already up
mkdir -p $C/results/awq; vlm $AB $AM $C/results/awq/vlm_16x 16; mark "AWQ_VLM16_EXIT $?"
# 2. AWQ with the card's headline vision settings (4x, max_slice_nums 36)
stop_vllm
LOG_TAG=_card4x EXTRA_ARGS="--mm-processor-kwargs {\"downsample_mode\":\"4x\",\"max_slice_nums\":36}" bash $C/scripts/serve_vllm_awq_5090.sh
wait_ready $AB $C/logs/vllm_awq_5090_18111.pid || mark AWQ_CARD_SERVER_FAILED
vlm $AB $AM $C/results/awq/vlm_4x_s36 16; mark "AWQ_VLM4X_EXIT $?"
stop_vllm
# 3. Q4 on the 5090: quality + speed
QB=http://127.0.0.1:18112/v1; QM=MiniCPM-V-4.7-35B-A3B-q4; R=$C/results/q4; mkdir -p $R
bash $C/scripts/serve_llama_gguf_5090.sh q4 18112
wait_ready $QB $C/logs/llama_q4_18112.pid || { mark Q4_SERVER_FAILED; }
nvidia-smi --query-gpu=index,memory.used --format=csv > $R/vram_5090_loaded.txt; mark Q4_READY
six_off $QB $QM $R/sixcat_strict_off; mark "Q4_SIXCAT_EXIT $?"
vlm $QB $QM $R/vlm_16x 16; mark "Q4_VLM_EXIT $?"
$VPY $C/scripts/needle_ladder.py --base-url $QB --model $QM --out $R/needle.json > $R/needle.stdout 2>&1; mark "Q4_NEEDLE_EXIT $?"
$VPY $C/scripts/decode_sweep.py --base-url $QB --model $QM --out $R/decode_sweep_5090.json > $R/decode_sweep.stdout 2>&1; mark "Q4_SWEEP_EXIT $?"
kill "$(cat $C/logs/llama_q4_18112.pid)"; sleep 8
# 4. Q3 speed on the 5090 (quality ran on the 3090)
R3=$C/results/q3; mkdir -p $R3
bash $C/scripts/serve_llama_gguf_5090.sh q3 18112
wait_ready $QB $C/logs/llama_q3_18112.pid || mark Q3_5090_SERVER_FAILED
nvidia-smi --query-gpu=index,memory.used --format=csv > $R3/vram_5090_loaded.txt
$VPY $C/scripts/decode_sweep.py --base-url $QB --model MiniCPM-V-4.7-35B-A3B-q3 --out $R3/decode_sweep_5090.json > $R3/decode_sweep.stdout 2>&1; mark "Q3_SWEEP_EXIT $?"
$VPY $C/scripts/needle_ladder.py --base-url $QB --model MiniCPM-V-4.7-35B-A3B-q3 --out $R3/needle_5090.json > $R3/needle_5090.stdout 2>&1; mark "Q3_NEEDLE5090_EXIT $?"
kill "$(cat $C/logs/llama_q3_18112.pid)"; sleep 8
# 5. llama-bench + wikitext-2 perplexity, both GGUFs, 5090
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
for A in Q4_K_M mix_q3gud_im; do
  $LLB/llama-bench -m $G/MiniCPM-V-4_7-35B-A3B-$A.gguf -ngl 999 -fa 1 -p 512,4096 -n 128 -r 3 -o json > $C/results/llama_bench_$A.json 2> $C/results/llama_bench_$A.stderr
  mark "BENCH_${A}_EXIT $?"
  $LLB/llama-perplexity -m $G/MiniCPM-V-4_7-35B-A3B-$A.gguf -f $C/fixtures/wikitext2_test.raw -ngl 999 -fa on -c 2048 --chunks 60 > $C/results/ppl_$A.txt 2>&1
  mark "PPL_${A}_EXIT $?"
done
# 6. restore the AWQ production server (as shipped)
LOG_TAG=_restored bash $C/scripts/serve_vllm_awq_5090.sh; wait_ready $AB $C/logs/vllm_awq_5090_18111.pid && mark AWQ_RESTORED || mark AWQ_RESTORE_FAILED
mark P5090_DONE
