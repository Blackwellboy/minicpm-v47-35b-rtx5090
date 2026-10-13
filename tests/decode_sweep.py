#!/usr/bin/env python3
"""Sustained-decode concurrency sweep: each request must emit exactly --out-tokens (ignore_eos), thinking off.
perf_probe.py's concurrency rows emit ~2 tokens per request, so their aggregate tok/s measures request overhead."""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

def req(base, model, i, n_out, prompt_words):
    body = {"model": model, "max_tokens": n_out, "temperature": 0.0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Request {i}. " + ("Write a long essay about GPUs. " + "lorem ipsum " * prompt_words)}]}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter(); p = json.loads(urllib.request.urlopen(r, timeout=900).read())
    return p["usage"]["completion_tokens"], p["usage"]["prompt_tokens"], time.perf_counter() - t

ap = argparse.ArgumentParser(); ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--out-tokens", type=int, default=512); ap.add_argument("--prompt-words", type=int, default=400)
a = ap.parse_args(); res = []
req(a.base_url, a.model, -1, 32, a.prompt_words)
for c in (1, 2, 4, 8, 16, 32, 64):
    t = time.perf_counter()
    with ThreadPoolExecutor(c) as ex:
        rs = list(ex.map(lambda i: req(a.base_url, a.model, i, a.out_tokens, a.prompt_words), range(c)))
    wall = time.perf_counter() - t; toks = sum(x[0] for x in rs)
    row = {"concurrency": c, "out_tokens_each": a.out_tokens, "prompt_tokens_each": rs[0][1], "wall_s": round(wall, 2),
           "aggregate_out_tps": round(toks / wall, 1), "per_req_tps_mean": round(sum(x[0] / x[2] for x in rs) / c, 1)}
    res.append(row); print(row, flush=True)
json.dump(res, open(a.out, "w"), indent=2)
