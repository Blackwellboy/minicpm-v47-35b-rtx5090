#!/usr/bin/env python3
"""Run CORRECTNESS_20-style capability pack against OpenAI-compatible chat."""
from __future__ import annotations
import argparse, json, re, subprocess, tempfile, textwrap, time, urllib.request
from pathlib import Path

def chat(base_url, model, prompt, max_tokens, temperature=0.0, enable_thinking=False, timeout=300):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    e2e = time.perf_counter() - t0
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    # strip empty think blocks if present
    if "</think>" in text:
        text = text.split("</think>")[-1].lstrip()
    return text, e2e, payload

def score_item(item, text):
    st = item["score"]["type"]
    expect = item["score"].get("expect")
    cleaned = text.strip()
    if st == "exact":
        # allow leading/trailing whitespace and optional quotes
        ok = cleaned == expect or cleaned.strip("`\"'") == expect
        return ok, cleaned
    if st == "contains":
        ok = expect in cleaned
        return ok, cleaned
    if st == "regex":
        ok = re.search(expect, cleaned, re.I | re.S) is not None
        return ok, cleaned
    if st == "python_exec_contains":
        # extract fenced or raw python
        m = re.search(r"```(?:python)?\n(.*?)```", cleaned, re.S)
        code = m.group(1) if m else cleaned
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sol.py"
            p.write_text(code)
            try:
                r = subprocess.run(["python3", str(p)], capture_output=True, text=True, timeout=10)
                out = (r.stdout or "") + (r.stderr or "")
                ok = item["score"]["expect_stdout"] in out and r.returncode == 0
                return ok, out[:500]
            except Exception as e:
                return False, repr(e)
    if st == "json_valid":
        try:
            json.loads(cleaned)
            return True, cleaned[:200]
        except Exception:
            # try extract json object
            m = re.search(r"\{.*\}", cleaned, re.S)
            if m:
                try:
                    json.loads(m.group(0))
                    return True, m.group(0)[:200]
                except Exception as e:
                    return False, repr(e)
            return False, cleaned[:200]
    return False, f"unknown_score_type:{st}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="MiniCPM5-2B")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()
    pack = json.loads(Path(args.pack).read_text())
    rows = []
    for item in pack["items"]:
        try:
            text, e2e, raw = chat(args.base_url, args.model, item["prompt"], item.get("max_tokens", 256), enable_thinking=args.enable_thinking)
            ok, detail = score_item(item, text)
            rows.append({"id": item["id"], "category": item["category"], "pass": ok, "e2e_s": e2e, "detail": detail, "text": text[:2000]})
            print(f"{item['id']} {'PASS' if ok else 'FAIL'} {e2e:.2f}s")
        except Exception as e:
            rows.append({"id": item["id"], "category": item["category"], "pass": False, "error": repr(e)})
            print(f"{item['id']} ERROR {e}")
    summary = {
        "pack_id": pack.get("pack_id"),
        "n": len(rows),
        "passed": sum(1 for r in rows if r.get("pass")),
        "by_category": {},
        "rows": rows,
    }
    for r in rows:
        c = r["category"]
        summary["by_category"].setdefault(c, {"n": 0, "pass": 0})
        summary["by_category"][c]["n"] += 1
        summary["by_category"][c]["pass"] += int(bool(r.get("pass")))
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("pack_id", "n", "passed", "by_category")}, indent=2))

if __name__ == "__main__":
    main()
