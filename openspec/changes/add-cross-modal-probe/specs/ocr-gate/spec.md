# Spec Delta

## Purpose

Verifies that the model under test can actually read the rendered images, so that a failed image probe is not mistaken for a statement about representations.

## ADDED Requirements

### Requirement: Transcription check with CER
The `ocr_check` stage SHALL ask the model to transcribe rendered clean content for a seeded sample of at least 30 items. It SHALL record each item's reference, transcription, and character error rate (CER).

#### Scenario: Per-item output
- **WHEN** `ocr_check` runs with `n_items = 40`
- **THEN** `ocr.jsonl` has 40 rows, each with `pair_id`, `reference`, `transcription`, and `cer`

### Requirement: Gate verdict
The stage SHALL write a summary with `n`, `median_cer`, and `pass`. `pass` SHALL be true only when `n >= 30` and the median CER is at most the configured gate (default 0.10).

#### Scenario: Gate passes
- **WHEN** 40 items have a median CER of 0.04
- **THEN** `ocr_summary.json` has `"pass": true`

#### Scenario: Too few items
- **WHEN** only 10 items are checked
- **THEN** `pass` is false and the summary records `reason: "n_below_minimum"`

### Requirement: Failed gate blocks image interpretation
When the OCR gate has not passed, every downstream image-condition result (P2, P3, image behavior) MUST be labeled "not interpretable (OCR)" in reports.

#### Scenario: Report after failed OCR
- **WHEN** `report` runs and `ocr_summary.json` has `"pass": false`
- **THEN** `headline.md` shows P2/P3 numbers with the label "not interpretable (OCR)" and does not classify an outcome
