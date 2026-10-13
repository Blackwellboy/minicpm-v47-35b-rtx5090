#!/usr/bin/env python3
"""Diagnostic (NOT a replacement score) for SIXCAT thinking-on runs.

Finds rows that finished with finish=stop, empty `content`, and a short answer sitting in `reasoning_content`.
That happens when the model answers without ever emitting </think>: the template pre-opens <think>, so the
qwen3 reasoning parser files the whole completion as reasoning. Reports what the category scores would be if the
first-letter / exact answer in reasoning were taken. The official SIXCAT score is the runner's, unchanged.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

log, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in log.open()][1:]
by = Counter(r["cat"] for r in rows); ok = Counter(r["cat"] for r in rows if r.get("ok"))
rec, misfiled = Counter(), []
for r in rows:
    if r.get("ok") or (r.get("raw_text") or "").strip() or r.get("finish") != "stop":
        continue
    rc = (r.get("reasoning_content") or "").strip()
    gold = str(r.get("gold")).strip()
    hit = False
    if r["cat"] in ("knowledge", "truth"):
        m = re.match(r"^\(?([A-J])\b", rc); hit = bool(m) and m.group(1) == gold
    elif len(rc) <= 40 and gold and gold.lower() in rc.lower():
        hit = True
    misfiled.append({"id": r.get("id"), "cat": r["cat"], "reasoning": rc[:120], "gold": gold, "would_pass": hit})
    rec[r["cat"]] += hit
cats = sorted(by)
official = {c: round(100 * ok[c] / by[c], 1) for c in cats}
recovered = {c: round(100 * (ok[c] + rec[c]) / by[c], 1) for c in cats}
res = {"source_log": log.name, "official_by_cat": official, "official_overall": round(sum(official.values()) / len(cats), 1),
       "if_reasoning_answers_taken_by_cat": recovered, "if_reasoning_answers_taken_overall": round(sum(recovered.values()) / len(cats), 1),
       "misfiled_rows": misfiled, "note": "diagnostic only; official score unchanged"}
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "misfiled_rows"}, indent=1))
