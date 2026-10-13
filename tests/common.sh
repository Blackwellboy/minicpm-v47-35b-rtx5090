# shared helpers for the MiniCPM-V 4.7 arm pipelines
C=$CAMPAIGN
VPY=$VENVS/vllm-minicpmv47/bin/python
QPY=$VENVS/vlmeval-quick/bin/python
SIX=$VENVS/sixcat050-a38d990/bin/sixcat
LLB=$SRC/llama.cpp-minicpmv47-20261013/build/bin
G=$MODELS/MiniCPM-V-4.7-35B-A3B-gguf
export SIXCAT_API_KEY=none HF_HUB_CACHE=$MODELS/datasets/hub HF_HUB_OFFLINE=1
mark() { echo "$1 $(date -Is)"; }
wait_ready() { # url pidfile
  for i in $(seq 1 240); do curl -sf "$1/models" >/dev/null && return 0; kill -0 "$(cat "$2")" 2>/dev/null || return 1; sleep 3; done; return 1; }
vlm() { # base model outdir conc [extra]
  $QPY -I $C/scripts/vlm_quickeval.py --base-url "$1" --model "$2" --bench all --n 250 --concurrency "$4" --thinking off --out-dir "$3" ${5:-} > "$3.stdout" 2>&1; }
six_off() { # base model outdir
  mkdir -p "$3"; $SIX --base-url "$1" --model "$2" --policy strict --thinking off --limit 20 --max-minutes 0 --request-timeout 1800 \
    --out "$3/sixcat.json" --log "$3/sixcat.jsonl" > "$3/runner.stdout" 2>&1; cp $C/fixtures/sixcat/SEALED_SIXCAT_STANDARD_V050.json "$3/"; }
