# Spec Delta

## Purpose

Provides a fast, sampled end-to-end run on the primary model, so the team gets an early signal check and something to present during the hackathon. It is clearly separated from the final results.

## ADDED Requirements

### Requirement: Demo stage
The `demo` stage SHALL sample N pairs from the `main` set (default 100, seed 42, stratified by `level`). It SHALL then run render → mini OCR check (default 10 items) → extract → present probes (LR + DoM) → report. Outputs SHALL be written to a `demo/` subfolder of the run.

#### Scenario: Demo outputs
- **WHEN** `demo` runs on the primary model config
- **THEN** `<run>/demo/reports/headline.md` and `<run>/demo/reports/layer_sweep_present.png` exist

#### Scenario: Stratified sample
- **WHEN** the main set has levels a, e, and g
- **THEN** the demo sample's per-level proportions match the main set's within one pair

### Requirement: Demo is informational, not a gate
Demo outputs SHALL be labeled as a signal check. The mini OCR check has fewer than 30 items and so cannot pass the OCR gate. Image-condition results in the demo MUST therefore be labeled "OCR not verified", and the demo MUST NOT classify an outcome.

#### Scenario: Demo labeling
- **WHEN** the demo headline is written
- **THEN** it states "demo – signal check, not a result", shows P2/P3 with "OCR not verified", and contains no outcome classification

### Requirement: Demo runtime
On one A100, the 100-pair demo on Qwen3.5-2B SHALL complete in under 15 minutes, excluding queue time and model download.

#### Scenario: Timed demo
- **WHEN** the demo runs in an interactive A100 session with the model cached
- **THEN** the wall time recorded in `run_manifest.json` is under 15 minutes
