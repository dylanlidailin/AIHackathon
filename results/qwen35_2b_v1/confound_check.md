# Confound check — /scratch/esd4uq/xmprobe/runs/qwen35_2b_v1

AUROC of each feature alone for clean (0) vs injected (1); 0.5 = no information, values far from 0.5 in either direction mean the feature separates the classes.

| Probe | Method | Layer | Feature | Feature-only AUROC | ρ(score, feature) clean | ρ injected |
|---|---|---|---|---|---|---|
| P1 | lr | 22 | chars | 0.465 | 0.15 | -0.11 |
| P1 | lr | 22 | image_height | 0.624 | 0.16 | -0.04 |
| P1 | dom | 19 | chars | 0.465 | -0.26 | -0.11 |
| P1 | dom | 19 | image_height | 0.624 | -0.20 | -0.01 |
| P2 | lr | 22 | chars | 0.465 | 0.34 | -0.06 |
| P2 | lr | 22 | image_height | 0.624 | 0.34 | -0.10 |
| P2 | dom | 19 | chars | 0.465 | -0.23 | -0.21 |
| P2 | dom | 19 | image_height | 0.624 | -0.25 | -0.19 |
| P3 | lr | 2 | chars | 0.465 | 0.43 | -0.20 |
| P3 | lr | 2 | image_height | 0.624 | 0.47 | -0.01 |
| P3 | dom | 20 | chars | 0.465 | -0.47 | -0.13 |
| P3 | dom | 20 | image_height | 0.624 | -0.40 | -0.06 |
