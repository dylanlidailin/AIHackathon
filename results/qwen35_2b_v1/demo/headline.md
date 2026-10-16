# Headline — qwen35_2b_v1 (demo)

> **demo – signal check, not a result.** Image-condition numbers are **OCR not verified** (mini OCR check, n=10 < 30). AUROC CIs at this sample size are roughly ±0.05–0.10.

## Scope

- Model: `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Renderer: DejaVuSans.ttf 22 px, width 1024, margin 32, line spacing 1.35
- Dataset: LLMail-Inject phase 1, scenario 1 — 100 main pairs, 0 probe_evading pairs (test-only)
- Splits: 5-fold grouped by `team_id`; AUROC fold-weighted; team bootstrap B = 1000

## Controls

- Length gap (injected − clean characters): main mean -4.6, max -82
- OCR gate: n=10, median CER 0.000, FAIL (gate ≤ 0.1, n ≥ 30)
- P1 gate (per method, AUROC ≥ 0.8): LR OCR not verified, DoM OCR not verified

## Outcome

No outcome classified: **demo – signal check, not a result**.

Headline layers (chosen on training-modality data only): LR P1/P2 L2, P3 L1; DoM P1/P2 L7, P3 L14

## Injection present — main

| Method | Probe | Layer | AUROC [95% CI] | Acc | Acc (centered) | n pos/neg | Status |
|---|---|---|---|---|---|---|---|
| LR | P1 | 2 | 1.000 [1.000, 1.000] | 0.990 | 0.990 | 100/100 |  |
| LR | P2 | 2 | 0.883 [0.815, 0.938] | 0.615 | 0.515 | 100/100 | OCR not verified |
| LR | P3 | 1 | 0.999 [0.996, 1.000] | 0.980 | 0.980 | 100/100 | OCR not verified |
| DoM | P1 | 7 | 1.000 [1.000, 1.000] | 0.965 | 0.965 | 100/100 |  |
| DoM | P2 | 7 | 0.970 [0.944, 0.991] | 0.500 | 0.500 | 100/100 | OCR not verified |
| DoM | P3 | 14 | 1.000 [1.000, 1.000] | 0.970 | 0.970 | 100/100 | OCR not verified |

Full layer sweep: `layer_sweep_present.png`.
