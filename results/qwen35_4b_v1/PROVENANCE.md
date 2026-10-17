# Provenance — qwen35_4b_v1

Run on Rivanna from a git checkout of this repository (the manifest records the commit of the last stage).

| Stage(s) | Code |
|---|---|
| render (login node), toolchain check, ocr_check, generate | `bc88d84`–`1e093a5` (src/ identical across these commits) |
| extract, probe + report | `1db20b3` (rerun after an out-of-memory fix: logits only for the last position; hidden states unchanged) |

- Hardware: NVIDIA RTX PRO 6000 Blackwell, MIG 1g.24gb slice, account `aikyamlab`, partition `gpu-mig`.
- Environment: `xmprobe` conda env; torch 2.8.0+cu128, torchvision 0.23.0, transformers 5.20.1.
- Data: same `data/pairs.jsonl` as qwen35_2b_v1 (sha256 prefix `5cb55f748e474eb9`); same render, probe, generate and OCR settings.
- The headline-layer rule (P1 ties broken by text d′) was fixed before this run.
