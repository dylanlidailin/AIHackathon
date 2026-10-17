# Cross-model comparison

Same pairs, renderer, probe settings and headline-layer rule for every model. Criteria C1–C5 were fixed before the Qwen3.5-4B and Qwen2.5-VL-3B runs (see the replicate-across-models spec).

| Model | Layers | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 24 | pass | pass | pass | pass | pass |
| Qwen3.5-4B | 32 | pass | pass | fail | pass | fail |
| Qwen2.5-VL-3B-Instruct | 36 | pass | fail | fail | fail | fail |

C1 late transfer · C2 threshold shift · C3 mid-depth dip · C4 direction alignment rises late · C5 obey does not transfer

## Shared pattern

- **C1: shared** by all models.
- C2: not shared: Qwen3.5-2B pass, Qwen3.5-4B pass, Qwen2.5-VL-3B-Instruct fail.
- C3: not shared: Qwen3.5-2B pass, Qwen3.5-4B fail, Qwen2.5-VL-3B-Instruct fail.
- C4: not shared: Qwen3.5-2B pass, Qwen3.5-4B pass, Qwen2.5-VL-3B-Instruct fail.
- C5: not shared: Qwen3.5-2B pass, Qwen3.5-4B fail, Qwen2.5-VL-3B-Instruct fail.

## Numbers behind each decision

**Qwen3.5-2B**
- C1: pass (LR L22 1.000 [1.000], DOM L19 0.999 [0.997])
- C2: pass (LR acc 0.50, DOM acc 0.50)
- C3: pass (LR mid-min 0.41, late-median 1.00; DOM mid-min 0.29, late-median 1.00)
- C4: pass (cos median middle 0.13, late 0.64 (gap +0.52))
- C5: pass (LR 0.41 [0.30, 0.59], DOM 0.57 [0.41, 0.78])

**Qwen3.5-4B**
- C1: pass (LR L17 1.000 [1.000], DOM L18 1.000 [1.000])
- C2: pass (LR acc 0.50, DOM acc 0.64)
- C3: fail (LR mid-min 1.00, late-median 1.00; DOM mid-min 0.99, late-median 0.99)
- C4: pass (cos median middle 0.52, late 0.81 (gap +0.29))
- C5: fail (LR 0.87 [0.75, 0.94], DOM 0.85 [0.72, 0.95])

**Qwen2.5-VL-3B-Instruct**
- C1: pass (LR L25 1.000 [1.000], DOM L35 0.998 [0.993])
- C2: fail (LR acc 0.99, DOM acc 0.91)
- C3: fail (LR mid-min 1.00, late-median 1.00; DOM mid-min 0.98, late-median 0.99)
- C4: fail (cos median middle 0.74, late 0.86 (gap +0.13))
- C5: fail (LR 0.91 [0.79, 1.00], DOM 0.96 [0.88, 1.00])

## Behavior

| Model | Group | Text target rate [Wilson] | Image target rate [Wilson] | McNemar p |
|---|---|---|---|---|
| Qwen3.5-2B | main | 0.156 [0.110, 0.218] | 0.162 [0.114, 0.224] | 1.000 |
| Qwen3.5-2B | probe_evading | 0.190 [0.109, 0.309] | 0.103 [0.048, 0.208] | 0.180 |
| Qwen3.5-4B | main | 0.220 [0.164, 0.287] | 0.092 [0.058, 0.145] | 0.001 |
| Qwen3.5-4B | probe_evading | 0.483 [0.359, 0.608] | 0.155 [0.084, 0.269] | 0.000 |
| Qwen2.5-VL-3B-Instruct | main | 0.457 [0.384, 0.531] | 0.052 [0.028, 0.096] | 0.000 |
| Qwen2.5-VL-3B-Instruct | probe_evading | 0.310 [0.206, 0.438] | 0.034 [0.010, 0.117] | 0.000 |

Figures: `p2_sweep_relative.png`, `direction_cosine_relative.png`.
