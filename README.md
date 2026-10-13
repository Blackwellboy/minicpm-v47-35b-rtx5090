# MiniCPM-V 4.7 35B-A3B on one RTX 5090

OpenBMB's MiniCPM-V 4.7 (35B total / 3B active MoE, vision + text, released 2026-10-11) tested on a single 32 GB RTX 5090
on 2026-10-13: three ways to fit it, the same tests on each, and the traps hit on the way.

| Arm | Weights | Size | Engine | VRAM loaded |
|---|---|---|---|---|
| **AWQ** | `openbmb/MiniCPM-V-4.7-35B-A3B-AWQ` (W4A16) | 22 GB | vLLM nightly | 26.3 GB (64K ctx, 139K-token KV pool) |
| **Q4_K_M** | `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` | 21.2 GB | llama.cpp | 25.4 GB (131K shared ctx) |
| **Q3 mix** | same repo, `mix_q3gud_im` (Q4_K_M with routed experts at Q3_K, imatrix) | 15.5 GB | llama.cpp | 20.0 GB (131K shared ctx) |

BF16 is 70.6 GB and FP8 about 36 GB, so neither fits one card.

## Headline

- **All three are usable.** The official Q3 mix scores within about 1 to 2 points of Q4_K_M and AWQ on every vision bench we ran, passes every long-context needle to 58K, and costs about 3% in wikitext-2 perplexity (6.04 vs 5.86). It frees 5 GB.
- **Speed:** one stream decodes at about 185 to 196 tok/s on every arm (llama-bench tg128: 231 Q4, 239 Q3). Under load vLLM pulls away: **2,883 tok/s aggregate at 32 concurrent** for AWQ vs a ~455 tok/s ceiling for the llama.cpp GGUFs.
- **Card check:** OCRBench reproduces (91.6 at the card's 4x/36-slice settings vs 91.2 claimed; 93 to 94 at 16x). MMStar, MMMU and MathVista land 7 to 13 points under the card on our n=250 rule-scored subsets, partly because the model writes long explanations even with thinking off (up to 51 of 250 MMMU items hit a 1,024-token cap). Read the card comparison with that in mind.
- **Use thinking OFF for direct answers.** With thinking on, short answers often never close the `<think>` block and come back as empty `content` (24 of 40 multiple-choice items). See [FINDINGS](docs/FINDINGS.md) and Minefield trap 162 (PR #203, pending).
- **The AWQ repo does not load as shipped** (stale remote-code pointers); one config overlay fixes it. Minefield trap 163 (PR #203, pending).
- **EXL3:** not possible with vision today. See [docs/EXL3.md](docs/EXL3.md).

## Results

Full tables: [results/SUMMARY.md](results/SUMMARY.md). Raw per-item rows for every bench are under `results/`.

### Vision subsets (n=250 per bench, thinking off, temperature 0, rule-based scoring)

| Bench | metric | AWQ 16x | AWQ 4x/36 | Q4_K_M 16x | Q3 mix 16x | card 16x | card 4x |
|---|---|---|---|---|---|---|---|
| ai2d | accuracy | 92.0 (2 cut) | 92.4 (1 cut) | 90.8 (6 cut) | 91.6 (5 cut) | - | - |
| chartqa | relaxed_acc@5% | 72.8 | 78.0 | 73.6 | 73.2 | - | - |
| docvqa | ANLS | 95.2 | 97.6 | 96.6 | 96.7 | - | - |
| hallusionbench | accuracy | 85.6 | 87.2 | 85.2 | 84.4 | 64.42 | 63.83 |
| mathvista | accuracy | 78.4 (30 cut) | 76.8 (32 cut) | 76.8 (26 cut) | 78.8 (23 cut) | 87.3 | 88.0 |
| mmmu | accuracy | 66.0 (48 cut) | 67.2 (45 cut) | 63.2 (51 cut) | 64.0 (48 cut) | 80.0 | 80.22 |
| mmstar | accuracy | 72.0 (1 cut) | 73.2 (1 cut) | 70.8 (1 cut) | 70.8 | 80.13 | 80.67 |
| ocrbench | accuracy | 93.2 | 91.6 | 94.0 (1 cut) | 93.6 | 90.4 | 91.2 |

### SIXCAT V050 (120 items, six categories)

| Arm | overall | knowledge | math | truth | instruct | code | tools |
|---|---|---|---|---|---|---|---|
| AWQ think off | 75.0 | 50.0 | 100.0 | 80.0 | 60.0 | 60.0 | 100.0 |
| AWQ think on | 64.2 | 25.0 | 100.0 | 25.0 | 80.0 | 70.0 | 85.0 |
| AWQ think on, card sampling | 65.8 | 40.0 | 100.0 | 20.0 | 70.0 | 80.0 | 85.0 |
| Q4_K_M think off | 78.3 | 60.0 | 100.0 | 80.0 | 85.0 | 50.0 | 95.0 |
| Q3 mix think off | 75.8 | 55.0 | 100.0 | 85.0 | 75.0 | 45.0 | 95.0 |

Thinking on, AWQ: 64.2 official; 78.3 if answers the parser filed as reasoning were counted (diagnostic, see trap 162).

### Decode throughput, RTX 5090 (512 forced output tokens, ~820-token prompt, thinking off)

| Arm | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
|---|---|---|---|---|---|---|---|
| AWQ (vLLM) | 184.7 | 277.2 | 566.4 | 1003.4 | 1767.2 | 2882.5 | 2370.8 |
| Q4_K_M (llama.cpp) | 196.3 | 329.2 | 396.8 | 439.1 | 455.1 | 362.5 | 357.2 |
| Q3 mix (llama.cpp) | 196.1 | 336.0 | 391.5 | 428.1 | 454.9 | 378.8 | 371.2 |

### llama-bench (5090, -fa 1, all layers on GPU) and wikitext-2 perplexity (ctx 2048, 60 chunks)

| GGUF | pp512 t/s | pp4096 t/s | tg128 t/s | PPL |
|---|---|---|---|---|
| Q4_K_M | 8085.9 | 7749.1 | 231.1 | 5.8612 ± 0.06102 |
| mix_q3gud_im | 7687.8 | 7188.4 | 239.0 | 6.0397 ± 0.06434 |

### Long-context passkey (10/50/90% depth at ~4K/16K/31K/58K real prompt tokens)

- AWQ: 12/12 (max 58403 prompt tokens)
- Q4_K_M: 12/12 (max 58403 prompt tokens)
- Q3 mix: 12/12 (max 58403 prompt tokens)


## Method notes

- Vision: 8 benches, n=250 each (first 250 rows), thinking off, temperature 0, scored from `content` only with rule-based extraction (no LLM judge). `(N cut)` = items that hit the token cap, counted wrong. HallusionBench here is per-question yes/no accuracy, **not** the card's aggregate, so the two are not comparable.
- SIXCAT 0.5.0 (sixcat-eval a38d990), strict policy, 120 items (20 per category). One item moves a category by 5 points, so overall gaps of about 3 points are noise. AWQ ran on vLLM, the GGUFs on llama.cpp: engine and quant differ together in that comparison.
- Throughput: 512 forced output tokens per request (`ignore_eos`), ~820-token prompt, thinking off. llama.cpp ran 16 slots with a unified KV pool.
- Perplexity: wikitext-2 test, ctx 2048, first 60 chunks, llama.cpp only (no AWQ equivalent).
- Everything on one RTX 5090 under WSL2, driver 610.62. Pins and hashes: [results/receipts](results/receipts).

## Run it yourself

[docs/SETUP.md](docs/SETUP.md) has both launch recipes. Tests live in [tests/](tests/).
