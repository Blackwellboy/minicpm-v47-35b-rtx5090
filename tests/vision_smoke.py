#!/usr/bin/env python3
"""Vision smoke for an OpenAI-compatible VLM: synthetic images with known answers (OCR, colour/shape, counting, chart read)."""
from __future__ import annotations
import argparse, base64, io, json, re, time, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default(size=sz)

def b64(img):
    buf = io.BytesIO(); img.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()

def cases():
    out = []
    im = Image.new("RGB", (640, 240), "white"); d = ImageDraw.Draw(im)
    d.text((40, 70), "INVOICE 48217-QX", fill="black", font=font(56))
    out.append(("ocr_code", im, "What exact code is written in the image? Reply with the text only.", r"48217-QX"))
    im = Image.new("RGB", (600, 300), "white"); d = ImageDraw.Draw(im)
    d.ellipse((40, 60, 220, 240), fill=(220, 20, 20)); d.rectangle((360, 70, 540, 250), fill=(20, 60, 220))
    out.append(("shape_colour", im, "What colour is the circle? Answer with one word.", r"\bred\b"))
    out.append(("shape_position", im, "Is the blue square on the left or the right? Answer with one word.", r"\bright\b"))
    im = Image.new("RGB", (700, 260), "white"); d = ImageDraw.Draw(im)
    for i in range(7):
        x = 30 + i * 95; d.polygon([(x, 200), (x + 40, 60), (x + 80, 200)], fill=(30, 150, 60))
    out.append(("count_triangles", im, "How many green triangles are in the image? Answer with a number only.", r"\b7\b|\bseven\b"))
    im = Image.new("RGB", (640, 400), "white"); d = ImageDraw.Draw(im)
    vals = {"Mon": 3, "Tue": 9, "Wed": 5, "Thu": 2}
    for i, (k, v) in enumerate(vals.items()):
        x = 80 + i * 130; d.rectangle((x, 340 - v * 30, x + 80, 340), fill=(90, 90, 200)); d.text((x + 10, 350), k, fill="black", font=font(28))
    out.append(("chart_max", im, "This bar chart has one bar per day. Which day has the tallest bar? Answer with the day label only.", r"\bTue"))
    return out

def chat(base, model, prompt, img, thinking, max_tokens):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(img)}},
                {"type": "text", "text": prompt}]}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        p = json.loads(r.read())
    return p, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    rows = []
    for cid, img, prompt, rx in cases():
        p, dt = chat(a.base_url, a.model, prompt, img, a.thinking, 4096 if a.thinking else 64)
        m = p["choices"][0]["message"]; text = (m.get("content") or "").strip()
        ok = re.search(rx, text, re.I) is not None
        rows.append({"id": cid, "pass": ok, "answer": text[:300], "finish": p["choices"][0].get("finish_reason"),
                     "prompt_tokens": p.get("usage", {}).get("prompt_tokens"), "e2e_s": round(dt, 3)})
        print(f"{cid} {'PASS' if ok else 'FAIL'} {dt:.2f}s -> {text[:80]!r}")
    s = {"n": len(rows), "passed": sum(r["pass"] for r in rows), "thinking": a.thinking, "rows": rows}
    Path(a.out).write_text(json.dumps(s, indent=2)); print(f"VISION {s['passed']}/{s['n']}")

if __name__ == "__main__":
    main()
