# Can MiniCPM-V 4.7 become an EXL3 (exllamav3) quant?

Short answer, checked 2026-10-13: **not as a vision model, today.**

- exllamav3 implements the text backbone family: `Qwen3_5MoeForConditionalGeneration` (with Qwen3-VL's vision tower) and
  `Qwen3NextForCausalLM`, including the Gated DeltaNet linear-attention layers. MiniCPM-V 4.7's language model uses the
  same layout (40 layers, 30 linear-attention + 10 full-attention, 256 experts / 8 active) and its weight names line up with
  the loader's `model.language_model.layers.N.*` pattern.
- There is **no MiniCPM-V architecture** in exllamav3: the `MiniCPMV4_7ForConditionalGeneration` wrapper and its SigLIP2
  vision tower (intra-ViT compression + merger) are not implemented. A converted file would be **text-only**.
- No MiniCPM-V 4.7 EXL3 exists on Hugging Face as of today. The backbone family has many (e.g. Qwen3.5/3.6-35B-A3B exl3).
- "rexl3": no format, fork or tool by that name turned up (Hugging Face, GitHub, web). The closest real things are vLLM
  EXL3 forks; none of them has MiniCPM-V vision plumbing either.

What a text-only attempt would take (untested, not done tonight): the 70.6 GB BF16 checkpoint, a retag of the architecture
to the Qwen3.5-MoE causal LM, conversion, then a KL / perplexity check against BF16, because MiniCPM's "canvas" M-RoPE
mode may not match Qwen's rope. It would lose vision, the thing this model is for. For a 32 GB card the official AWQ
(vLLM) and Q3/Q4 GGUFs (llama.cpp) already cover it with vision intact.
