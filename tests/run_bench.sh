#!/usr/bin/env bash
# Bench pass for MiniCPM-V-4.7-35B-A3B-AWQ on RTX 5090 (vLLM nightly, port 18111).
set -uo pipefail
C="$(cd "$(dirname "$0")/.." && pwd)"
B=http://127.0.0.1:18111/v1; M=MiniCPM-V-4.7-35B-A3B-AWQ
PY=$VENVS/vllm-minicpmv47/bin/python
SIX=$VENVS/sixcat050-a38d990/bin/sixcat
export SIXCAT_API_KEY=none
R=$C/results; mkdir -p $R/perf $R/capability $R/sixcat_strict_off $R/sixcat_strict_on
echo "BENCH_START $(date -Is)"
$PY $C/scripts/perf_probe.py --base-url $B --model $M --out $R/perf/perf_5090.json > $R/perf/stdout.txt 2>&1; echo "PERF_EXIT $? $(date -Is)"
$PY $C/scripts/run_capability_pack.py --base-url $B --model $M --pack $C/fixtures/capability/CORRECTNESS_20.json --out $R/capability/capability_5090.json > $R/capability/stdout.txt 2>&1; echo "CAP_EXIT $? $(date -Is)"
for TH in off on; do
  O=$R/sixcat_strict_$TH
  $SIX --base-url $B --model $M --policy strict --thinking $TH --limit 20 --max-minutes 0 --request-timeout 1800 \
    --out $O/sixcat.json --log $O/sixcat.jsonl > $O/runner.stdout 2>&1
  echo "SIXCAT_${TH}_EXIT $? $(date -Is)"
done
cp $C/fixtures/sixcat/SEALED_SIXCAT_STANDARD_V050.json $R/sixcat_strict_off/ ; cp $C/fixtures/sixcat/SEALED_SIXCAT_STANDARD_V050.json $R/sixcat_strict_on/
echo "BENCH_DONE $(date -Is)"
