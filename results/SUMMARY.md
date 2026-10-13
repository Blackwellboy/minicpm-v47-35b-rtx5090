## Vision subsets (n=250 per bench, thinking off, temperature 0, rule-based scoring)

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

## SIXCAT V050 (120 items, six categories)

| Arm | overall | knowledge | math | truth | instruct | code | tools |
|---|---|---|---|---|---|---|---|
| AWQ think off | 75.0 | 50.0 | 100.0 | 80.0 | 60.0 | 60.0 | 100.0 |
| AWQ think on | 64.2 | 25.0 | 100.0 | 25.0 | 80.0 | 70.0 | 85.0 |
| AWQ think on, card sampling | 65.8 | 40.0 | 100.0 | 20.0 | 70.0 | 80.0 | 85.0 |
| Q4_K_M think off | 78.3 | 60.0 | 100.0 | 80.0 | 85.0 | 50.0 | 95.0 |
| Q3 mix think off | 75.8 | 55.0 | 100.0 | 85.0 | 75.0 | 45.0 | 95.0 |

Thinking on, AWQ: 64.2 official; 78.3 if answers the parser filed as reasoning were counted (diagnostic, see trap 162).

## Decode throughput, RTX 5090 (512 forced output tokens, ~820-token prompt, thinking off)

| Arm | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
|---|---|---|---|---|---|---|---|
| AWQ (vLLM) | 184.7 | 277.2 | 566.4 | 1003.4 | 1767.2 | 2882.5 | 2370.8 |
| Q4_K_M (llama.cpp) | 196.3 | 329.2 | 396.8 | 439.1 | 455.1 | 362.5 | 357.2 |
| Q3 mix (llama.cpp) | 196.1 | 336.0 | 391.5 | 428.1 | 454.9 | 378.8 | 371.2 |

## llama-bench (5090, -fa 1, all layers on GPU) and wikitext-2 perplexity (ctx 2048, 60 chunks)

| GGUF | pp512 t/s | pp4096 t/s | tg128 t/s | PPL |
|---|---|---|---|---|
| Q4_K_M | 8085.9 | 7749.1 | 231.1 | 5.8612 ± 0.06102 |
| mix_q3gud_im | 7687.8 | 7188.4 | 239.0 | 6.0397 ± 0.06434 |

## Long-context passkey (10/50/90% depth at ~4K/16K/31K/58K real prompt tokens)

- AWQ: 12/12 (max 58403 prompt tokens)
- Q4_K_M: 12/12 (max 58403 prompt tokens)
- Q3 mix: 12/12 (max 58403 prompt tokens)
