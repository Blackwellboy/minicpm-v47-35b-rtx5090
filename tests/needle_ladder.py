#!/usr/bin/env python3
"""Passkey retrieval ladder: one 7-digit key hidden at 10/50/90% depth in varied filler, thinking off.
Reports realized prompt_tokens from the server (requested length is not proof of realized length)."""
import argparse, json, random, time, urllib.request
from pathlib import Path

TOPICS = ["river", "harbour", "orchard", "library", "glacier", "market", "observatory", "railway", "vineyard", "lighthouse"]
VERBS = ["described", "measured", "recorded", "painted", "mapped", "photographed", "catalogued", "studied"]

def filler(rng, n_sent):
    return " ".join(f"In {1700 + rng.randrange(300)} a traveller {rng.choice(VERBS)} the {rng.choice(TOPICS)} near the {rng.choice(TOPICS)} and wrote {rng.randrange(2, 90)} pages about it."
                    for _ in range(n_sent))

def ask(base, model, prompt, timeout=900):
    body = {"model": model, "temperature": 0.0, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter(); p = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    return p, time.perf_counter() - t

ap = argparse.ArgumentParser(); ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--targets", default="4000,16000,32000,60000")
a = ap.parse_args(); rows = []
for target in [int(x) for x in a.targets.split(",")]:
    for depth in (0.1, 0.5, 0.9):
        rng = random.Random(target * 10 + int(depth * 10)); key = str(rng.randrange(1_000_000, 9_999_999))
        n = max(4, int(target / 24.5))  # ~24.5 tokens per filler sentence
        sents = filler(rng, n).split(". ")
        k = int(len(sents) * depth)
        sents.insert(k, f"IMPORTANT: the secret passkey is {key}. Remember it")
        prompt = ". ".join(sents) + "\n\nWhat is the secret passkey? Reply with the digits only."
        p, dt = ask(a.base_url, a.model, prompt)
        ans = (p["choices"][0]["message"].get("content") or "").strip()
        row = {"target": target, "depth": depth, "prompt_tokens": p["usage"]["prompt_tokens"], "pass": key in ans,
               "answer": ans[:40], "e2e_s": round(dt, 2), "prefill_tps_proxy": round(p["usage"]["prompt_tokens"] / dt, 0)}
        rows.append(row); print(row, flush=True)
s = {"passed": sum(r["pass"] for r in rows), "n": len(rows), "max_prompt_tokens": max(r["prompt_tokens"] for r in rows), "rows": rows}
Path(a.out).write_text(json.dumps(s, indent=2)); print(f"NEEDLE {s['passed']}/{s['n']} max_ptok={s['max_prompt_tokens']}")
