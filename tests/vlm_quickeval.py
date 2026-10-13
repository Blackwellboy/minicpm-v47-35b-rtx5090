#!/usr/bin/env python3
"""vlm_quickeval -- fast, engine-neutral VLM benchmark subset harness.

Targets any OpenAI-compatible /v1/chat/completions endpoint (vLLM, llama.cpp
llama-server with --mmproj, SGLang, ...). Images are sent inline as
data:image/{png,jpeg};base64 URIs in `image_url` content parts; thinking is
toggled with `chat_template_kwargs: {"enable_thinking": bool}` (accepted by
both vLLM and llama-server).

Datasets are read as pinned parquet shards from the HF hub cache (data only:
no dataset scripts, no trust_remote_code). Only the rows needed are decoded.

Scoring is OFFICIAL-FROM-CONTENT: `pred`/`correct` come from message.content
only (a closed <think>...</think> block inside content is stripped first).
If content is empty, the item is scored wrong; we additionally report
`would_pass_from_reasoning` = items whose reasoning text (reasoning_content /
reasoning / an unclosed <think> in content) contains a passing answer.
That count is diagnostic only and never folded into accuracy.

Subset caveats (also written into summary.json):
  * AI2D uses only test shard 0 of 2 (1544 of 3088 rows).
  * DocVQA uses only validation shard 0 of 6 (892 rows); n is capped at 300.
  * MMMU: multiple-choice items only (open-ended skipped).
  * HallusionBench: image split only (951 q); metric = per-question aAcc.
  * MathVista: rule-based extraction only (no GPT judge as in the official
    pipeline); expect a few points below judge-scored numbers.
  * Official numbers use the full sets and often VLMEvalKit's LLM-judge
    fallback for MC extraction; these are rule-only subsets.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

MC_SUFFIX = "Answer with the option's letter from the given choices directly."
SHORT_SUFFIX = "Answer the question using a single word or phrase."
LETTERS = "ABCDEFGHIJ"

# Pinned dataset revisions (verified 2026-10-13 via HF API).
BENCHES = {
    "mmstar": dict(repo="Lin-Chen/MMStar", rev="bc98d668301da7b14f648724866e57302778ab27",
                   files=["mmstar.parquet"], split="val", cap=None),
    "ai2d": dict(repo="lmms-lab/ai2d", rev="c83a9b9692933aff8349157c88a413df9d02c4e5",
                 files=["data/test-00000-of-00002.parquet"], split="test (shard 0/2)", cap=None),
    "mmmu": dict(repo="lmms-lab/MMMU", rev="364f2e2eb107b36e07ff4c5a15f5947a759cef47",
                 files=["data/validation-00000-of-00001.parquet"], split="validation (MC only)", cap=None),
    "ocrbench": dict(repo="echo840/OCRBench", rev="92a54bd1384387c178d5a07140a2d85e0a3d12e1",
                     files=["data/test-00000-of-00001.parquet"], split="test", cap=None),
    "chartqa": dict(repo="lmms-lab/ChartQA", rev="9e63b7df1592a1c2158e735cc1725454aef0d6d9",
                    files=["data/test-00000-of-00001.parquet"], split="test", cap=None),
    "hallusionbench": dict(repo="lmms-lab/HallusionBench", rev="cd417161857aefb23d878d42cf1bb53aa9dd646f",
                           files=["data/image-00000-of-00001.parquet"], split="image", cap=None),
    "mathvista": dict(repo="AI4Math/MathVista", rev="2b6ad69445fbb5695c9b165475e8decdbeb97747",
                      files=["data/testmini-00000-of-00001-725687bf7a18d64b.parquet"], split="testmini", cap=None),
    "docvqa": dict(repo="lmms-lab/DocVQA", rev="539088ef8a8ada01ac8e2e6d4e372586748a265e",
                   files=["DocVQA/validation-00000-of-00006.parquet"], split="validation (shard 0/6)", cap=300),
}
BENCH_ORDER = list(BENCHES)
LONG_BENCHES = {"mathvista", "mmmu"}
CAVEATS = [
    "AI2D: test shard 0 of 2 only (1544 of 3088 rows).",
    "DocVQA: validation shard 0 of 6 only (892 rows), capped at 300.",
    "MMMU: multiple-choice items only; open-ended skipped.",
    "HallusionBench: image split only; metric is per-question aAcc.",
    "MathVista: rule-based extraction, no GPT judge; expect a few points below judge-scored numbers.",
    "All MC extraction is rule-only (no VLMEvalKit LLM-judge fallback); subsets, not full sets.",
]


# --------------------------------------------------------------------------- data

def shard_path(repo, rev, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", revision=rev)


def image_cols(schema):
    out = []
    for f in schema:
        t = str(f.type)
        if f.name in ("image", "decoded_image") or re.fullmatch(r"image_\d+", f.name):
            if "binary" in t:
                out.append(f.name)
    return out


def load_rows(bench, n, sample, seed, eligible_fn):
    """Return (rows, meta). Reads light columns for every row (cheap) to apply
    eligibility, picks indices, then decodes only the row groups holding them."""
    cfg = BENCHES[bench]
    pfs = [pq.ParquetFile(shard_path(cfg["repo"], cfg["rev"], f)) for f in cfg["files"]]
    pf = pfs[0]  # single shard per bench by design
    icols = image_cols(pf.schema_arrow)
    light = [c for c in pf.schema_arrow.names if c not in icols]
    light_tbl = pf.read(columns=light).to_pylist()
    eligible = [i for i, r in enumerate(light_tbl) if eligible_fn(r)]
    if cfg["cap"]:
        eligible = eligible[: cfg["cap"]]
    if n is None or n >= len(eligible):
        picked = list(eligible)
    elif sample == "first":
        picked = eligible[:n]
    else:
        picked = sorted(random.Random(seed).sample(eligible, n))
    # map global row index -> (row group, offset)
    bounds, start = [], 0
    for g in range(pf.num_row_groups):
        cnt = pf.metadata.row_group(g).num_rows
        bounds.append((start, start + cnt))
        start += cnt
    need = {}
    for idx in picked:
        for g, (a, b) in enumerate(bounds):
            if a <= idx < b:
                need.setdefault(g, []).append(idx - a)
                break
    rows_by_idx = {}
    for g, offs in need.items():
        t = pf.read_row_group(g, columns=icols)
        for o in offs:
            rows_by_idx[bounds[g][0] + o] = t.slice(o, 1).to_pylist()[0]
    rows = []
    for idx in picked:
        r = dict(light_tbl[idx])
        r.update(rows_by_idx[idx])
        r["_row"] = idx
        rows.append(r)
    meta = dict(repo=cfg["repo"], revision=cfg["rev"], files=cfg["files"], split=cfg["split"],
                total_rows=pf.metadata.num_rows, eligible_rows=len(eligible), cap=cfg["cap"])
    return rows, meta


def to_data_uri(img):
    """img: struct {bytes,path} or raw bytes -> data URI. PNG/JPEG passed through
    untouched; anything else re-encoded to PNG via Pillow."""
    b = img.get("bytes") if isinstance(img, dict) else img
    if b is None:
        raise ValueError("image has no bytes")
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "png"
    elif b[:3] == b"\xff\xd8\xff":
        mime = "jpeg"
    else:
        from PIL import Image
        im = Image.open(io.BytesIO(b))
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b, mime = buf.getvalue(), "png"
    return f"data:image/{mime};base64," + base64.b64encode(b).decode()


def img_part(img):
    return {"type": "image_url", "image_url": {"url": to_data_uri(img)}}


def mc_block(question, options):
    lines = [question.strip(), "Options:"]
    lines += [f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options)]
    lines.append(MC_SUFFIX)
    return "\n".join(lines)


def parse_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    try:
        x = ast.literal_eval(v)  # literal parser only, never executes code
        return list(x) if isinstance(x, (list, tuple)) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- bench adapters
# Each adapter: eligible(row)->bool, build(row)->item dict with
#   id, question(text), gold, content(list of parts), kind, extra

def ad_mmstar():
    def build(r):
        q = r["question"]
        # options are embedded: "Options: A: ..., B: ..., C: ..., D: ..."
        opts = {}
        m = re.search(r"Options:\s*(.*)$", q, re.S)
        if m:
            parts = re.split(r"(?:^|,\s*)([A-J]):\s*", m.group(1))
            for i in range(1, len(parts) - 1, 2):
                opts[parts[i]] = parts[i + 1].strip().rstrip(".").strip()
        stem = q[: m.start()].strip() if m else q.strip()
        if opts:
            text = mc_block(stem, [opts[k] for k in sorted(opts)])
        else:
            text = q.strip() + "\n" + MC_SUFFIX
        return dict(id=f"mmstar-{r['index']}", question=text, gold=r["answer"].strip().upper(),
                    content=[img_part(r["image"]), {"type": "text", "text": text}],
                    kind="mc", options=[opts[k] for k in sorted(opts)] if opts else None,
                    extra=dict(category=r["category"]))
    return (lambda r: True), build


def ad_ai2d():
    def build(r):
        opts = parse_list(r["options"])
        text = mc_block(r["question"], opts)
        return dict(id=f"ai2d-{r['_row']}", question=text, gold=LETTERS[int(r["answer"])],
                    content=[img_part(r["image"]), {"type": "text", "text": text}],
                    kind="mc", options=opts, extra={})
    return (lambda r: True), build


def ad_mmmu():
    def eligible(r):
        return r["question_type"] == "multiple-choice" and bool(parse_list(r["options"]))

    def build(r):
        opts = parse_list(r["options"])
        text = mc_block(r["question"], opts)
        imgs = {k: r.get(f"image_{k}") for k in range(1, 8)}
        imgs = {k: v for k, v in imgs.items() if v and v.get("bytes")}
        pieces = re.split(r"(<image \d+>)", text)
        referenced = [int(re.findall(r"\d+", p)[0]) for p in pieces if re.fullmatch(r"<image \d+>", p)]
        content = []
        # images never referenced by placeholder go first
        for k in sorted(imgs):
            if k not in referenced:
                content.append(img_part(imgs[k]))
        for p in pieces:
            mm = re.fullmatch(r"<image (\d+)>", p)
            if mm and int(mm.group(1)) in imgs:
                content.append(img_part(imgs[int(mm.group(1))]))
            elif p:
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] += p
                else:
                    content.append({"type": "text", "text": p})
        return dict(id=r["id"], question=text, gold=r["answer"].strip().upper(), content=content,
                    kind="mc", options=opts, extra=dict(n_images=len(imgs), subfield=r["subfield"]))
    return eligible, build


def ad_ocrbench():
    def build(r):
        q = r["question"]
        golds = parse_list(r["answer"]) or [str(r["answer"])]
        return dict(id=f"ocrbench-{r['_row']}", question=q, gold=golds,
                    content=[img_part(r["image"]), {"type": "text", "text": q}],
                    kind="ocrbench", extra=dict(dataset=r["dataset"], question_type=r["question_type"]))
    return (lambda r: True), build


def ad_chartqa():
    def build(r):
        q = r["question"].strip() + "\n" + SHORT_SUFFIX
        return dict(id=f"chartqa-{r['_row']}", question=q, gold=str(r["answer"]),
                    content=[img_part(r["image"]), {"type": "text", "text": q}],
                    kind="relaxed", extra=dict(type=r["type"]))
    return (lambda r: True), build


def ad_hallusion():
    def eligible(r):
        return str(r.get("visual_input")) != "0"

    def build(r):
        q = r["question"].strip() + " Please answer yes or no."
        gold = "yes" if str(r["gt_answer"]).strip() == "1" else "no"
        rid = f"hallusion-{r['category']}-{r['subcategory']}-{r['set_id']}-{r['figure_id']}-{r['question_id']}"
        return dict(id=rid, question=q, gold=gold,
                    content=[img_part(r["image"]), {"type": "text", "text": q}],
                    kind="yorn", extra=dict(category=r["category"], subcategory=r["subcategory"]))
    return eligible, build


def ad_mathvista():
    def build(r):
        q = r["query"]
        choices = parse_list(r["choices"]) if r["choices"] is not None else None
        return dict(id=f"mathvista-{r['pid']}", question=q, gold=str(r["answer"]),
                    content=[img_part(r["decoded_image"]), {"type": "text", "text": q}],
                    kind="mathvista", options=choices,
                    extra=dict(question_type=r["question_type"], answer_type=r["answer_type"],
                               precision=r["precision"]))
    return (lambda r: True), build


def ad_docvqa():
    def build(r):
        q = r["question"].strip() + "\n" + SHORT_SUFFIX
        return dict(id=f"docvqa-{r['questionId']}", question=q, gold=list(r["answers"]),
                    content=[img_part(r["image"]), {"type": "text", "text": q}],
                    kind="anls", extra={})
    return (lambda r: True), build


ADAPTERS = dict(mmstar=ad_mmstar, ai2d=ad_ai2d, mmmu=ad_mmmu, ocrbench=ad_ocrbench, chartqa=ad_chartqa,
                hallusionbench=ad_hallusion, mathvista=ad_mathvista, docvqa=ad_docvqa)


# --------------------------------------------------------------------------- extraction / scoring

THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S)


def split_content(content):
    """-> (answer_text, leaked_reasoning). Strips closed think blocks; an unclosed
    <think> (or a stray </think>) means everything before/after is reasoning."""
    c = content or ""
    leaked = ""
    for m in THINK_CLOSED.finditer(c):
        leaked += m.group(0)
    c = THINK_CLOSED.sub("", c)
    if "<think>" in c:
        i = c.index("<think>")
        leaked += c[i:]
        c = c[:i]
    if "</think>" in c:  # template opened <think> in the prompt
        i = c.index("</think>")
        leaked += c[:i]
        c = c[i + len("</think>"):]
    return c.strip(), leaked


def extract_letter(text, options=None, n_opts=None, last=False):
    if not text:
        return None
    valid = LETTERS[: (n_opts or (len(options) if options else 10))]
    t = text.strip()
    # 1. bare letter / "(B)" / "B." / "B:" at the very start
    m = re.match(r"^\(?\s*([A-J])\s*(?:[\).:,]|$|\s)", t)
    if m and m.group(1) in valid and not last:
        return m.group(1)
    long = last or len(t) > 40  # CoT-style output: trust the end, not the start
    # 2. explicit "answer is X" style (letter must be uppercase)
    ms = list(re.finditer(r"(?i:answer|option|choice)\s*(?i:is)?\s*[:\-]?\s*[\*\(\[]*\s*([A-J])(?![A-Za-z])", t))
    ms = [x for x in ms if x.group(1) in valid]
    if ms:
        return (ms[-1] if long else ms[0]).group(1)
    standalone = r"(?<![A-Za-z0-9'])([A-J])(?![A-Za-z0-9'])"
    if long:
        lines = [l for l in t.splitlines() if l.strip()]
        tail = lines[-1] if lines else ""
        ms = [x for x in re.finditer(standalone, tail) if x.group(1) in valid]
        if ms:
            return ms[-1].group(1)
        if options:
            hits = [i for i, o in enumerate(options) if str(o).strip() and str(o).strip().lower() in tail.lower()]
            if len(hits) == 1:
                return LETTERS[hits[0]]
        return None  # long output with no answer at the end: unscorable, not guessed
    # 3. first standalone capital letter in the valid set
    ms = [x for x in re.finditer(standalone, t) if x.group(1) in valid]
    if ms:
        return ms[0].group(1)
    # 4. option text match (unique)
    if options:
        low = t.lower()
        hits = [i for i, o in enumerate(options) if str(o).strip() and str(o).strip().lower() in low]
        if len(hits) == 1:
            return LETTERS[hits[0]]
    return None


def to_float(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def relaxed_correct(pred, gold, tol=0.05):
    p, g = to_float(pred), to_float(gold)
    if p is not None and g is not None:
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= tol
    return str(pred).strip().lower().rstrip(".") == str(gold).strip().lower()


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, golds, thr=0.5):
    p = " ".join(str(pred).strip().lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(str(g).strip().lower().split())
        d = levenshtein(p, g)
        ln = max(len(p), len(g)) or 1
        s = 1 - d / ln
        best = max(best, s if s >= thr else 0.0)
    return best


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def final_line(text):
    """Short-answer extraction from a multi-line (explain-then-answer) reply:
    the last non-empty line, with markdown bold / 'Answer:' prefix / trailing
    period removed. Single-line replies are returned stripped."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    t = lines[-1].replace("**", "").strip()
    t = re.sub(r"^(?:final answer|answer)\s*[:\-]\s*", "", t, flags=re.I)
    return t.rstrip(".").strip()


def score(item, text, from_reasoning=False):
    """-> (pred, score in [0,1]). text = answer text (content) or reasoning."""
    kind = item["kind"]
    last = from_reasoning or ("\n" in (text or "").strip()) or len((text or "").strip()) > 40
    if kind == "mc":
        p = extract_letter(text, item.get("options"), last=last)
        return p, float(p is not None and p == item["gold"])
    if kind == "yorn":
        ms = re.findall(r"\b(yes|no)\b", final_line(text).lower()) or re.findall(r"\b(yes|no)\b", (text or "").lower())
        p = (ms[-1] if last else ms[0]) if ms else None
        return p, float(p == item["gold"])
    if kind == "ocrbench":
        t = text or ""
        if item["extra"]["dataset"] == "HME100k":
            pr = t.strip().replace("\n", " ").replace(" ", "")
            ok = any(str(g).strip().replace("\n", " ").replace(" ", "") in pr for g in item["gold"])
        else:
            pr = t.lower().strip().replace("\n", " ")
            ok = any(str(g).lower().strip().replace("\n", " ") in pr for g in item["gold"])
        return t.strip()[:200], float(ok and bool(t.strip()))
    if kind == "relaxed":
        t = (text or "").strip()
        if from_reasoning:
            nums = NUM_RE.findall(t)
            cands = [nums[-1]] if nums else []
            cands.append(t.splitlines()[-1] if t else "")
            ok = any(relaxed_correct(c, item["gold"]) for c in cands)
            return (cands[0] if cands else None), float(ok)
        t = final_line(t)
        return t, float(bool(t) and relaxed_correct(t, item["gold"]))
    if kind == "anls":
        t = (text or "").strip()
        t = final_line(t)
        return t, (anls(t, item["gold"]) if t else 0.0)
    if kind == "mathvista":
        t = (text or "").strip()
        ex = item["extra"]
        if ex["question_type"] == "multi_choice":
            opts = item.get("options") or []
            letter = extract_letter(t, opts, last=last)
            p = opts[LETTERS.index(letter)] if letter and LETTERS.index(letter) < len(opts) else None
            if p is None:  # direct option text
                low = t.lower()
                hits = [o for o in opts if str(o).strip().lower() in low]
                p = hits[0] if len(hits) == 1 else None
            return p, float(p is not None and str(p).strip() == item["gold"].strip())
        nums = NUM_RE.findall(t)
        if not nums:
            return None, 0.0
        raw = nums[-1] if (last or len(nums) > 1) else nums[0]
        p = to_float(raw)
        g = to_float(item["gold"])
        if p is None or g is None:
            return raw, float(raw.strip() == item["gold"].strip())
        if ex["answer_type"] == "integer":
            return raw, float(round(p) == p and int(p) == int(g))
        prec = ex.get("precision")
        if prec is not None:
            return raw, float(round(p, int(prec)) == round(g, int(prec)))
        return raw, float(abs(p - g) <= 1e-6 * max(1.0, abs(g)))
    raise ValueError(kind)


def selftest():
    mc = dict(kind="mc", gold="B", options=["cat", "dog", "bird", "fish"])
    assert score(mc, "B")[1] == 1 and score(mc, "(B)")[1] == 1 and score(mc, "B. dog")[1] == 1
    assert score(mc, "A")[1] == 0 and score(mc, "dog")[1] == 1
    assert score(mc, "Looking at the picture carefully, I see a dog.\nThe answer is B.")[1] == 1
    assert score(mc, "Let me compute: option A gives 3, which is wrong here, so we")[1] == 0  # truncated CoT
    assert score(mc, "so the final answer is **B**", from_reasoning=True)[1] == 1
    assert score(mc, "Step 1: B is the vertex." + chr(10) + "Now we compute the angle")[1] == 0  # truncated
    assert score(dict(kind="relaxed", gold="3"), "We count Mauritania, Fiji and Madagascar.\n\n3")[1] == 1
    assert score(dict(kind="anls", gold=["itc limited"]), "The top-left says ITC Limited 2013.\n**ITC Limited**")[1] == 1
    yn = dict(kind="yorn", gold="no")
    assert score(dict(kind="yorn", gold="yes"), "Is there no doubt? Checking the chart again.\nYes")[1] == 1
    assert score(yn, "No.")[1] == 1 and score(yn, "Yes")[1] == 0
    cq = dict(kind="relaxed", gold="0.57")
    assert score(cq, "0.57")[1] == 1 and score(cq, "0.6")[1] == 0 and score(cq, "0.58")[1] == 1
    assert score(dict(kind="relaxed", gold="Yes"), "yes")[1] == 1
    assert score(dict(kind="relaxed", gold="62"), "62%")[1] == 1
    oc = dict(kind="ocrbench", gold=["CENTRE"], extra=dict(dataset="IIIT5K"))
    assert score(oc, "Centre")[1] == 1 and score(oc, "Centrie")[1] == 0
    hm = dict(kind="ocrbench", gold=["x ^ { 2 }"], extra=dict(dataset="HME100k"))
    assert score(hm, "x ^ {2}")[1] == 1 and score(hm, "X^{2}")[1] == 0
    dv = dict(kind="anls", gold=["university of california"])
    assert score(dv, "University of California")[1] == 1.0 and score(dv, "univ")[1] == 0.0
    mv = dict(kind="mathvista", gold="1.2", options=None,
              extra=dict(question_type="free_form", answer_type="float", precision=1.0))
    assert score(mv, "so d = 1.2")[1] == 1 and score(mv, "1.3")[1] == 0
    mvc = dict(kind="mathvista", gold="145°", options=["135°", "140°", "145°", "150°"],
               extra=dict(question_type="multi_choice", answer_type="text", precision=None))
    assert score(mvc, "C")[1] == 1 and score(mvc, "A")[1] == 0
    assert split_content("<think>x is B</think>A") == ("A", "<think>x is B</think>")
    assert split_content("<think>unclosed B")[0] == ""
    print("selftest OK")


# --------------------------------------------------------------------------- client

def call(base_url, payload, timeout):
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_item(args, item, max_tokens):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["content"]}],
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    if args.seed_request is not None:
        payload["seed"] = args.seed_request
    rec = {k: item[k] for k in ("id", "question", "gold")}
    rec["extra"] = item.get("extra", {})
    t0 = time.time()
    err = None
    resp = None
    for attempt in range(args.retries + 1):
        try:
            resp = call(args.base_url, payload, args.timeout)
            err = None
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}"
            if e.code < 500:
                break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(1 + attempt)
    rec["latency_s"] = round(time.time() - t0, 3)
    if err is not None:
        rec.update(error=err, content=None, reasoning_content=None, reasoning=None, finish_reason=None,
                   prompt_tokens=None, completion_tokens=None, pred=None, correct=False, score=0.0,
                   empty_content=None, would_pass_from_reasoning=False)
        return rec
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some servers return parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    usage = resp.get("usage") or {}
    rec.update(error=None, content=content, reasoning_content=msg.get("reasoning_content"),
               reasoning=msg.get("reasoning"), finish_reason=ch.get("finish_reason"),
               prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
    answer, leaked = split_content(content)
    rec["answer_text"] = answer
    rec["empty_content"] = not answer
    pred, sc = score(item, answer) if answer else (None, 0.0)
    rec["pred"], rec["score"] = pred, sc
    rec["correct"] = sc >= 1.0 if item["kind"] != "anls" else sc > 0
    wp = False
    if not answer:
        rtext = "\n".join(x for x in (msg.get("reasoning_content"), msg.get("reasoning"), leaked) if x)
        if rtext:
            _, rs = score(item, rtext, from_reasoning=True)
            wp = rs >= (1.0 if item["kind"] != "anls" else 0.5)
    rec["would_pass_from_reasoning"] = wp
    return rec


# --------------------------------------------------------------------------- driver

def bench_max_tokens(args, bench):
    if args.max_tokens is not None:
        return args.max_tokens
    if args.thinking == "on":
        return 4096
    # MathVista's official query asks for reasoning + final value, and the model
    # does CoT on hard MMMU items even when told to answer directly; 64 tokens
    # would truncate nearly every item, so these get a longer budget.
    return args.long_max_tokens if bench in LONG_BENCHES else 64


def run_bench(args, bench, out_root):
    out = os.path.join(out_root, bench)
    os.makedirs(out, exist_ok=True)
    items_path = os.path.join(out, "items.jsonl")
    eligible, build = ADAPTERS[bench]()
    t_load = time.time()
    rows, meta = load_rows(bench, args.n, args.sample, args.seed, eligible)
    done = {}
    if args.resume and os.path.exists(items_path):
        with open(items_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done[r["id"]] = r
    items = []
    for r in rows:
        try:
            items.append(build(r))
        except Exception as e:  # noqa: BLE001
            print(f"[{bench}] build failed row {r.get('_row')}: {e}", file=sys.stderr)
    load_s = time.time() - t_load
    todo = [it for it in items if it["id"] not in done]
    mt = bench_max_tokens(args, bench)
    print(f"[{bench}] {len(items)} items ({len(done)} resumed), load {load_s:.1f}s", flush=True)
    results = dict(done)
    lock = threading.Lock()
    t0 = time.time()
    with open(items_path, "w") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_item, args, it, mt): it for it in todo}
            for k, fu in enumerate(as_completed(futs), 1):
                rec = fu.result()
                with lock:
                    results[rec["id"]] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if k % 25 == 0 or k == len(todo):
                    print(f"[{bench}] {k}/{len(todo)}", flush=True)
    wall = time.time() - t0
    recs = [results[it["id"]] for it in items if it["id"] in results]
    ok = [r for r in recs if r.get("error") is None]
    n = len(recs)
    lat = [r["latency_s"] for r in ok]
    summ = dict(
        bench=bench, metric={"anls": "ANLS", "relaxed": "relaxed_acc@5%"}.get(items[0]["kind"] if items else "", "accuracy"),
        n=n, n_scored_ok=len(ok), errors=n - len(ok),
        accuracy=round(sum(r["score"] for r in recs) / n, 4) if n else None,
        correct=sum(1 for r in recs if r.get("correct")),
        empty_content=sum(1 for r in ok if r.get("empty_content")),
        would_pass_from_reasoning=sum(1 for r in ok if r.get("would_pass_from_reasoning")),
        finish_length=sum(1 for r in ok if r.get("finish_reason") == "length"),
        mean_latency_s=round(sum(lat) / len(lat), 3) if lat else None,
        mean_completion_tokens=round(sum(r["completion_tokens"] or 0 for r in ok) / len(ok), 1) if ok else None,
        wall_time_s=round(wall, 2), load_time_s=round(load_s, 2),
        selection=dict(mode=args.sample, n_requested=args.n, seed=args.seed if args.sample == "seeded" else None,
                       row_indices=[r["_row"] for r in rows]),
        dataset=meta,
        request=dict(base_url=args.base_url, model=args.model, thinking=args.thinking,
                     temperature=args.temperature, max_tokens=mt, concurrency=args.concurrency),
        scoring_note="official score from message.content only; would_pass_from_reasoning is diagnostic",
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required="--selftest" not in sys.argv, help="e.g. http://127.0.0.1:18111/v1")
    ap.add_argument("--model", required="--selftest" not in sys.argv)
    ap.add_argument("--bench", default="all", help="all or comma list of: " + ",".join(BENCH_ORDER))
    ap.add_argument("--n", type=int, default=None, help="items per bench (default: all eligible, docvqa capped 300)")
    ap.add_argument("--sample", choices=["first", "seeded"], default="first",
                    help="first = deterministic first-N eligible rows; seeded = random.Random(seed).sample")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="force one value for every bench. Default: thinking on 4096; thinking off 64, "
                         "except mathvista/mmmu which use --long-max-tokens")
    ap.add_argument("--long-max-tokens", type=int, default=1024,
                    help="thinking-off budget for mathvista and mmmu when --max-tokens is not given")
    ap.add_argument("--seed-request", type=int, default=None, help="optional 'seed' field sent to the server")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="reuse error-free items already in items.jsonl")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true", help="run offline scorer tests and exit")
    if "--selftest" in sys.argv:
        selftest()
        return
    args = ap.parse_args()
    if not args.out_dir:
        ap.error("--out-dir is required")
    benches = BENCH_ORDER if args.bench == "all" else [b.strip() for b in args.bench.split(",")]
    for b in benches:
        if b not in BENCHES:
            ap.error(f"unknown bench {b}")
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    summaries = {}
    for b in benches:
        summaries[b] = run_bench(args, b, args.out_dir)
        s = summaries[b]
        print(f"[{b}] {s['metric']}={s['accuracy']} n={s['n']} err={s['errors']} empty={s['empty_content']} "
              f"wpr={s['would_pass_from_reasoning']} len={s['finish_length']} lat={s['mean_latency_s']}s "
              f"wall={s['wall_time_s']}s", flush=True)
    combined = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        argv=sys.argv, total_wall_time_s=round(time.time() - t0, 2),
        table={b: {k: s[k] for k in ("metric", "accuracy", "n", "errors", "empty_content",
                                     "would_pass_from_reasoning", "finish_length", "mean_latency_s", "wall_time_s")}
               for b, s in summaries.items()},
        caveats=CAVEATS,
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(json.dumps(combined["table"], indent=1))


if __name__ == "__main__":
    main()
