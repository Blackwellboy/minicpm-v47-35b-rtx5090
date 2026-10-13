#!/usr/bin/env python3
"""Append-only adjudication for run_capability_pack.py output.

run_capability_pack.py (inherited from MINICPM5_2B_3090_5090_20260908) has no scorer for score types
`json_exact` / `json_schema`, so those items always FAIL with `unknown_score_type`. This re-scores only those
rows from the stored model text (JSON-equal to `expect`; json_schema = same keys, values and types) and leaves
every other verdict untouched. The raw file is not modified.
"""
import json, re, sys
from pathlib import Path

def parse(text):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)

def main(pack_path, raw_path, out_path):
    pack = {i["id"]: i for i in json.loads(Path(pack_path).read_text())["items"]}
    raw = json.loads(Path(raw_path).read_text())
    rows, changed = [], []
    for r in raw["rows"]:
        r = dict(r)
        st = pack[r["id"]]["score"]["type"]
        if st in ("json_exact", "json_schema") and str(r.get("detail", "")).startswith("unknown_score_type"):
            try:
                ok = parse(r.get("text", "")) == pack[r["id"]]["score"]["expect"]
            except Exception as e:
                ok = False
            r["adjudicated_pass"] = ok
            changed.append({"id": r["id"], "raw_pass": r["pass"], "adjudicated_pass": ok, "reason": f"missing scorer for {st}"})
        else:
            r["adjudicated_pass"] = bool(r.get("pass"))
        rows.append(r)
    out = {"raw_file": Path(raw_path).name, "raw_passed": raw["passed"], "n": len(rows),
           "adjudicated_passed": sum(r["adjudicated_pass"] for r in rows), "changes": changed}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(*sys.argv[1:4])
