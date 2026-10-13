# MiniCPM-V 4.7 35B-A3B on one RTX 5090

Test notebook for OpenBMB's MiniCPM-V 4.7 (35B total, 3B active, vision + text) on a single 32 GB RTX 5090.

Status: **in progress (2026-10-13)**. Private until reviewed.

Arms:

| Arm | Weights | Engine |
|---|---|---|
| AWQ | `openbmb/MiniCPM-V-4.7-35B-A3B-AWQ` (W4A16, 22 GB) | vLLM nightly |
| Q4 | `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` Q4_K_M (21.2 GB) | llama.cpp |
| Q3 | `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` mix_q3gud_im (15.5 GB) | llama.cpp |

Layout: `tests/` runnable harness, `results/` per-arm outputs, `docs/` setup notes and traps, `post/` write-up.
