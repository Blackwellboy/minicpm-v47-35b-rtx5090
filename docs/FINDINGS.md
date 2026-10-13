# Findings and traps from the bring-up

## 1. Thinking mode hides short answers (Minefield trap 162, PR #203 pending)

The chat template opens `<think>\n` whenever thinking is on (the default). On short multiple-choice prompts the model often
answers immediately, e.g. `B` then end-of-turn, and never writes `</think>`. vLLM's `--reasoning-parser qwen3` then files the
whole completion as reasoning: `content: null`, the letter sits in `reasoning`, `finish_reason: stop`.

Measured on SIXCAT V050 (AWQ, vLLM): 24 of 40 knowledge + truth items came back with empty content at temperature 0, and
24 of 40 again at the model card's sampling (temperature 1.0, top_p 0.95, top_k 20). 17 of the temperature-0 ones were the
right letter. Official score 64.2; 78.3 if those answers were counted. With thinking off: 0 of 40 empty, score 75.0.

What to do: send `"chat_template_kwargs": {"enable_thinking": false}` (or `--reasoning off` on llama.cpp) for direct
answers, vision Q&A and tool calls. Keep thinking for math, code and long instruction tasks, where it helped
(SIXCAT instruct 60 -> 80, code 60 -> 70/80).

## 2. The official AWQ repo does not load as shipped (Minefield trap 163, PR #203 pending)

`openbmb/MiniCPM-V-4.7-35B-A3B-AWQ` @ `ea491d0` keeps `auto_map` entries for `processing_minicpmv4_7.py`,
`image_processing_minicpmv4_7.py` and `video_processing_minicpmv4_7.py`, none of which are in the repo, plus
`image_processor_type: MiniCPMV4_7ImageProcessor`, which transformers 5.19 does not have. vLLM refuses without
`--trust-remote-code`; stripping `auto_map` then fails with "Unrecognized image processor". The base repo moved to native
transformers configs on 2026-10-13. Fix: `tests/make_native_overlay.py` (base configs over AWQ weights). Chat template and
tokenizer are byte-identical between the two repos.

## 3. Hybrid model needs `--max-num-seqs` below the state-cache size

30 of 40 layers are Gated DeltaNet (one recurrent state block per sequence). With the defaults vLLM stops at startup:
`max_num_seqs (256) exceeds available Mamba cache blocks (217)`. `--max-num-seqs 64` fixes it; throughput peaks at c32 anyway.

## 4. Visible explanations even with thinking off

With thinking off the model still explains before answering on MMMU and MathVista: 30 to 48 of 250 items per bench hit a
1,024-token cap. Those count as wrong in our tables. The card's eval budget is not stated; give direct-answer prompts more
room than 1,024 tokens if you want card-like scores.

## 5. Metric mismatches against the card

- HallusionBench: our number is per-question yes/no accuracy. The card's 63.8/64.4 is a different aggregate. Not comparable.
- MMMU / MathVista: our scoring is rule-based (no LLM answer extractor) on n=250 subsets. Expect a few points low.
- OCRBench matches the card (91.6 at the card's 4x/36-slice settings vs 91.2 claimed).

## 6. Other bring-up notes

- vLLM support is nightly-only (PR #58674). llama.cpp needs PR #29416 or later; older GGUFs compute M-RoPE wrong.
- The template emits Qwen3-Coder XML tool calls: `--tool-call-parser qwen3_coder`. Tool smoke: exact args, `finish_reason: tool_calls`.
- `hf download` with several `--include` patterns dropped the first one on our box (Q4_K_M never arrived). Check files after download.
- llama.cpp under WSL on the second card (RTX 3090) refused a 14.5 GB weight allocation with 23 GB reported free, after one
  successful load. Not chased; every number here is from the 5090.
