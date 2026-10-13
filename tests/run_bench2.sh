#!/usr/bin/env bash
# Pass 2: SIXCAT thinking-on with the model card's generation_config sampling (temp 1.0, top_p 0.95, top_k 20, seed 1),
# then a sustained-decode concurrency sweep. Pass 1 (run_bench.sh) found strict temp-0 thinking-on answers filed as reasoning.
set -uo pipefail
C="$(cd "$(dirname "$0")/.." && pwd)"
B=http://127.0.0.1:18111/v1; M=MiniCPM-V-4.7-35B-A3B-AWQ
PY=$VENVS/vllm-minicpmv47/bin/python
SIX=$VENVS/sixcat050-a38d990/bin/sixcat
export SIXCAT_API_KEY=none
R=$C/results; O=$R/sixcat_cardsampling_on; mkdir -p $O $R/decode_sweep
echo "BENCH2_START $(date -Is)"
$SIX --base-url $B --model $M --policy custom --temperature 1.0 --top-p 0.95 --top-k 20 --seed 1 --thinking on \
  --limit 20 --max-minutes 0 --request-timeout 1800 --out $O/sixcat.json --log $O/sixcat.jsonl > $O/runner.stdout 2>&1
echo "SIXCAT_card_on_EXIT $? $(date -Is)"
cp $C/fixtures/sixcat/SEALED_SIXCAT_STANDARD_V050.json $O/
$PY $C/scripts/decode_sweep.py --base-url $B --model $M --out $R/decode_sweep/decode_sweep_5090.json > $R/decode_sweep/stdout.txt 2>&1
echo "SWEEP_EXIT $? $(date -Is)"
nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu --format=csv -i 0 >> $R/decode_sweep/stdout.txt
echo "BENCH2_DONE $(date -Is)"
