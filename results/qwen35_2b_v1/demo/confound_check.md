# Confound check — /scratch/esd4uq/xmprobe/runs/qwen35_2b_v1/demo

AUROC of each feature alone for clean (0) vs injected (1); 0.5 = no information, values far from 0.5 in either direction mean the feature separates the classes.

| Probe | Method | Layer | Feature | Feature-only AUROC | ρ(score, feature) clean | ρ injected |
|---|---|---|---|---|---|---|
| P1 | lr | 2 | chars | 0.471 | -0.12 | 0.06 |
| P1 | lr | 2 | image_height | 0.636 | -0.17 | 0.13 |
| P1 | dom | 7 | chars | 0.471 | -0.15 | -0.15 |
| P1 | dom | 7 | image_height | 0.636 | -0.24 | -0.02 |
| P2 | lr | 2 | chars | 0.471 | -0.43 | -0.23 |
| P2 | lr | 2 | image_height | 0.636 | -0.46 | -0.16 |
| P2 | dom | 7 | chars | 0.471 | 0.04 | 0.11 |
| P2 | dom | 7 | image_height | 0.636 | 0.07 | 0.11 |
| P3 | lr | 1 | chars | 0.471 | 0.38 | -0.10 |
| P3 | lr | 1 | image_height | 0.636 | 0.44 | 0.03 |
| P3 | dom | 14 | chars | 0.471 | 0.34 | -0.28 |
| P3 | dom | 14 | image_height | 0.636 | 0.34 | -0.13 |
