# Post draft (not posted)

## Thread

1/ MiniCPM-V 4.7 dropped this week: 35B-A3B MoE, vision + text, OpenBMB claims it beats Qwen3.6-35B-A3B on most vision benches. I put it on one RTX 5090 tonight three ways (AWQ on vLLM, official Q4_K_M and Q3 GGUFs on llama.cpp) and ran the same tests on all three. Repo: <link>

2/ Fit: BF16 is 70.6 GB, so it's a quant or nothing on 32 GB. AWQ = 22 GB, Q4_K_M = 21.2 GB, the official "mix_q3gud_im" Q3 = 15.5 GB (Q4_K_M with only the routed experts at Q3_K). All three load with vision and leave room for 64K to 131K of context.

3/ The Q3 is the surprise. Across 8 vision benches (n=250 each) it lands within 1 to 2 points of Q4_K_M and AWQ. 12/12 passkeys to 58K tokens. Wikitext-2 perplexity 6.04 vs 5.86 for Q4 (+3%). OpenBMB said it "matches Q4_K_M on benchmarks"; on ours, it does.

4/ Speed on one 5090: a single stream decodes at ~185 to 196 tok/s on every arm (llama-bench tg128: 231 Q4, 239 Q3). Under load vLLM + AWQ pulls away: 2,883 tok/s aggregate at 32 concurrent requests vs ~455 tok/s for llama.cpp. Prefill ~13K tok/s at 58K context on vLLM.

5/ Card check: OCRBench reproduces (91.6 at the card's 4x settings vs 91.2 claimed). MMStar / MMMU / MathVista come in 7 to 13 points under the card on our rule-scored subsets, partly because the model explains at length even with thinking off. Not a debunk; a "your harness matters" note.

6/ Trap 1: leave thinking OFF for direct answers. With thinking on, short answers often never close the <think> block, so vLLM's reasoning parser files the answer as reasoning and content comes back empty. 24 of 40 multiple-choice items on our text suite. Submitted to the Model Serving Minefield as trap 162 (PR #203).

7/ Trap 2: the official AWQ repo doesn't load as shipped. Its configs point at remote-code files that aren't in the repo; the base repo moved to native transformers configs on release day. Copy the base repo's 3 processor/tokenizer configs over the AWQ weights and it runs. Submitted as trap 163.

8/ EXL3? Not with vision today: exllamav3 supports the Qwen3.5-MoE backbone but not MiniCPM's SigLIP2 vision tower. A text-only conversion is possible in principle and pointless for a VLM.

9/ Recipes, every raw row, pins and hashes: <link>. vLLM nightly, llama.cpp master, one 5090, no cherry-picked runs (a failed Q4 pass is documented, not hidden).

## One-paragraph version

MiniCPM-V 4.7 35B-A3B runs well on a single RTX 5090. The official AWQ on vLLM nightly (after a three-file config fix) decodes ~185 tok/s per stream and 2,883 tok/s aggregate at 32 concurrent; the official GGUFs on llama.cpp match it per stream (~196 tok/s) but top out around 455 tok/s aggregate. The 15.5 GB Q3 mix scores within 1 to 2 points of Q4_K_M on eight vision benches and costs 3% perplexity. OCRBench reproduces the card; MMStar/MMMU/MathVista run 7 to 13 points under it on our subsets. Keep thinking off for direct answers: with it on, short answers vanish into an unclosed think block.
