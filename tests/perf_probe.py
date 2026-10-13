#!/usr/bin/env python3
"""Bench-style latency/throughput probe for OpenAI-compatible chat."""
from __future__ import annotations
import argparse, json, os, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def post_chat(base_url, model, messages, max_tokens, temperature, seed=None, timeout=600):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
    # Prefer thinking off for latency baselines unless requested
    body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        t1 = time.perf_counter()
        payload = json.loads(raw)
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    return {
        "e2e_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "text": text,
        "raw": payload,
    }

def one_short(base_url, model):
    return post_chat(
        base_url, model,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=8, temperature=0.0, seed=1,
    )

def one_long_decode(base_url, model, out_tokens=1024):
    # ~1k-ish input filler + forced long output
    filler = ("The quick brown fox jumps over the lazy dog. " * 40)
    prompt = (
        "Write a continuous numbered list from 1 to 400 with one number per line. "
        "Do not stop early. Start now.\n\nContext filler:\n" + filler
    )
    return post_chat(
        base_url, model,
        [{"role": "user", "content": prompt}],
        max_tokens=out_tokens, temperature=0.0, seed=1,
    )

def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)

def concurrency_run(base_url, model, conc, n_requests, max_tokens=64):
    def work(i):
        return post_chat(
            base_url, model,
            [{"role": "user", "content": f"Say the integer {i} and nothing else."}],
            max_tokens=max_tokens, temperature=0.0, seed=i,
        )
    t0 = time.perf_counter()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(work, i) for i in range(n_requests)]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(repr(e))
    wall = time.perf_counter() - t0
    e2e = [r["e2e_s"] for r in results]
    outs = [r["completion_tokens"] or 0 for r in results]
    aggregate_out_tps = (sum(outs) / wall) if wall > 0 else None
    per_req_tps = []
    for r in results:
        ct = r["completion_tokens"] or 0
        if r["e2e_s"] > 0 and ct:
            per_req_tps.append(ct / r["e2e_s"])
    return {
        "concurrency": conc,
        "n_requests": n_requests,
        "ok": len(results),
        "errors": errors,
        "wall_s": wall,
        "aggregate_out_tps": aggregate_out_tps,
        "e2e_p50": pct(e2e, 50),
        "e2e_p95": pct(e2e, 95),
        "per_req_tps_p50": pct(per_req_tps, 50),
        "completion_tokens_sum": sum(outs),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="MiniCPM5-2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()
    report = {"base_url": args.base_url, "model": args.model, "ts": time.time()}
    # warm
    try:
        report["warm"] = one_short(args.base_url, args.model)
    except Exception as e:
        report["warm_error"] = repr(e)
        Path = __import__("pathlib").Path
        Path(args.out).write_text(json.dumps(report, indent=2))
        raise
    report["c1_short"] = one_short(args.base_url, args.model)
    if not args.skip_long:
        report["c1_long_decode"] = one_long_decode(args.base_url, args.model)
    report["concurrency"] = []
    for c in (1, 2, 4, 8, 16):
        n = max(c * 2, 8)
        try:
            report["concurrency"].append(concurrency_run(args.base_url, args.model, c, n))
        except Exception as e:
            report["concurrency"].append({"concurrency": c, "fatal": repr(e)})
            break
    from pathlib import Path
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k not in ("warm",)}, indent=2)[:4000])

if __name__ == "__main__":
    main()
