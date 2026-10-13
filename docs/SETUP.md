# Setup: three ways to run MiniCPM-V 4.7 35B-A3B on one RTX 5090

Hardware: RTX 5090 32 GB under WSL2 (Ubuntu), NVIDIA driver 610.62. All numbers in this repo come from that one card.

## AWQ on vLLM (fastest, best concurrency)

- Weights: `openbmb/MiniCPM-V-4.7-35B-A3B-AWQ` @ `ea491d0` (compressed-tensors W4A16, vision tower unquantized, 22 GB).
- Engine: vLLM nightly `0.31.1rc1.dev371+gdf6c7ad03` (MiniCPM-V 4.7 support is main-only, vLLM PR #58674), torch 2.13 cu130, transformers 5.19.
  `pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly`
- **Config overlay is required** (see trap 163): the AWQ repo still points `auto_map` at remote-code files it does not ship.
  `python tests/make_native_overlay.py <awq_dir> <base_repo_configs_dir> <overlay_dir>` symlinks the AWQ weights and copies
  `processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json` from `openbmb/MiniCPM-V-4.7-35B-A3B` @ `17e6511`.
  No `--trust-remote-code`, no remote code executed.
- Launch: `tests/serve_vllm_awq_5090.sh`

```
--max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192
--max-num-seqs 64          # default 256 > 217 hybrid (GDN) state blocks: startup refuses
--limit-mm-per-prompt '{"image": 5, "video": 1}'
--reasoning-parser qwen3
--enable-auto-tool-choice --tool-call-parser qwen3_coder   # template emits <tool_call><function=...>
env CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 + tests/vllm_wsl_patch (WSL pin-memory)
```

Card vision settings (4x token compression, 36 slices) are a server flag:
`--mm-processor-kwargs '{"downsample_mode":"4x","max_slice_nums":36}'` (default is 16x / 9 slices).

Backends selected: Marlin WNA16 MoE, FlashAttention (text and ViT), CUDA graphs FULL_AND_PIECEWISE. 21.1 GiB weights,
2.8 GiB KV = 139,328 tokens of cache, ~26.3 GB VRAM idle.

## GGUF on llama.cpp (Q4_K_M and the official Q3 mix)

- Weights: `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` @ `1e04eae`: `Q4_K_M` (21.2 GB) and `mix_q3gud_im` (15.5 GB; Q4_K_M with
  only the routed-expert gate/up/down at Q3_K, imatrix), projector `mmproj-model-f16.gguf` (16x).
- Engine: llama.cpp master built today with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120"` (needs PR #29416 or later;
  older converters compute M-RoPE wrong).
- Launch: `tests/serve_llama_gguf_5090.sh q4|q3`

```
-ngl 999 -fa on -c 131072 --parallel 16 --kv-unified --jinja --reasoning-format deepseek
--temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.0   # CookBook uses 1.05; 1.0 here so the quant comparison matches vLLM
```

Thinking is ON by default in the chat template. Send `"chat_template_kwargs": {"enable_thinking": false}` per request
(or `--reasoning off` on llama.cpp) for direct answers. See trap 162 for why that matters.

## Tests

| Script | What |
|---|---|
| `tests/vlm_quickeval.py` | 8 vision benches, n=250 each, OpenAI endpoint, engine-neutral, rule-based scoring |
| SIXCAT 0.5.0 (`sixcat-eval` a38d990), `tests/fixtures/SEALED_SIXCAT_STANDARD_V050.json` | 6-category text suite, 120 items |
| `tests/needle_ladder.py` | passkey at 10/50/90% depth, 4K to 58K real prompt tokens |
| `tests/decode_sweep.py` | sustained decode, 512 forced tokens, c1 to c64 |
| `tests/run_capability_pack.py` + `tests/adjudicate_capability.py` | 26-item correctness pack |
| `tests/vision_smoke.py`, `tests/smoke_text_tools.py` | bring-up smoke |
| `tests/pipe_*.sh` | the exact orchestration used tonight |
