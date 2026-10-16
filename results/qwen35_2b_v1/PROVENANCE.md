# Provenance — qwen35_2b_v1

`run_manifest.json` records `git_commit: null` because the code reached Rivanna by rsync
(without `.git`). The code each stage ran is recorded here instead.

| Stage(s) | SLURM job | When (2026-10-16, EDT) | Code |
|---|---|---|---|
| render | login node | 15:06 | `f199e1e` |
| ocr_check (n = 40) | 21461543 | 15:07–15:08 | `f199e1e` |
| generate | 21461544 | 15:08–15:20 | `f199e1e` |
| extract | 21461545 | 15:20–15:23 | `f199e1e` |
| probe + report (first pass, first-index layer choice) | 21461546 | 15:23–15:30 | `f199e1e` |
| probe + report (final, d′ tie-break) | 21462560 | 15:35–15:41 | `949cd98` |
| report (y-axis fix) | login node | 15:43 | `949cd98` |
| demo (in `demo/`) | 21461247 | 14:57–15:00 | `3e4855a` + the edits committed in `f199e1e` |

- Hardware: NVIDIA RTX PRO 6000 Blackwell (MIG 1g.24gb slice), account `aikyamlab`, partition `gpu-mig`.
  The planned account `ai-safety-hackathon-rig` had no SLURM association on 2026-10-16.
- Environment: `xmprobe` conda env (clone of `torch311`), torch 2.8.0+cu128, torchvision 0.23.0,
  transformers 5.20.1, huggingface_hub 1.33.0.
- Model: `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Data: `data/pairs.jsonl` sha256 prefix `5cb55f748e474eb9` (173 main / 58 probe_evading).
- The d′ tie-break for the headline layer was added after seeing the first pass; see `headline.md`.
- Next runs should use a `git clone` on Rivanna so the manifest records the commit.
