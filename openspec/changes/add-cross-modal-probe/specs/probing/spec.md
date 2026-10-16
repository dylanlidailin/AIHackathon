# Spec Delta

## Purpose

Trains and evaluates linear probes that test whether an injection signal learned from text inputs transfers to image inputs. It reports both co-primary methods, with leakage-free cross-validation and calibrated uncertainty.

## ADDED Requirements

### Requirement: Three probe roles per layer
For each layer the system SHALL compute three probes. P1 is fit on text and scored on held-out text. P2 is the same text-fit probe, scored on held-out image pairs. P3 is fit on images and scored on held-out images.

#### Scenario: P2 reuses the text fit
- **WHEN** fold k is evaluated
- **THEN** the P2 scores for fold k come from the exact model or direction fitted for P1 in fold k

### Requirement: Two co-primary methods
Every probe SHALL be computed with two methods: logistic regression (LR) and difference of means (DoM). DoM uses direction = mean(injected) − mean(clean) over the training-modality training pairs, and its score is the projection onto that direction. DoM has no fitted hyperparameters. Both methods MUST be reported for every probe.

#### Scenario: Both methods in output
- **WHEN** `probe` completes
- **THEN** `probes/present.csv` has rows with `method = lr` and rows with `method = dom` for every probe × layer

#### Scenario: DoM uses training folds only
- **WHEN** the DoM direction is computed for a fold
- **THEN** it uses only that fold's training `team_id` groups from the training modality (text for P1/P2, image for P3)

### Requirement: Grouped outer cross-validation with pooled scores
Probes SHALL use 5-fold cross-validation grouped by `team_id` (seed 42). Out-of-fold scores SHALL be pooled so that each `main` pair is scored exactly once per probe, method, and layer.

#### Scenario: Pooled coverage
- **WHEN** scoring finishes
- **THEN** every `main` pair has exactly one out-of-fold score per probe × method × layer

### Requirement: Hyperparameters and layer chosen from the training modality only
The LR regularization strength SHALL be chosen by inner grouped cross-validation on the training modality only. The headline layer SHALL be chosen separately for each method: for P1/P2 it is the argmax of that method's P1 out-of-fold text AUROC, and for P3 it comes from that method's inner image CV. Image data MUST NOT influence any P2 choice.

#### Scenario: No image peeking for P2
- **WHEN** the P2 headline layer is selected
- **THEN** the selection uses only text-modality scores, and the chosen layer is recorded per method

### Requirement: AUROC is primary, with bootstrap intervals
AUROC SHALL be the primary metric, with a 95% interval from a group-level bootstrap (B = 1000). Accuracy SHALL be reported as secondary, at a threshold chosen on training-modality data.

#### Scenario: Interval columns
- **WHEN** results are written
- **THEN** each row has `auroc`, `ci_lo`, `ci_hi`, and `acc`

### Requirement: Centered variant reports accuracy only
A per-modality-centered variant (subtract the modality's mean clean activation, computed on training pairs) SHALL be reported for both methods, for accuracy only. Centering shifts every score within a modality by a constant, so it cannot change AUROC. The output MUST NOT present a centered AUROC as a separate result.

#### Scenario: Centered rows
- **WHEN** the centered variant is computed
- **THEN** its rows report `acc` and leave `auroc` empty or equal to the uncentered value, with a note

### Requirement: Present and obey targets
The `present` target SHALL label clean = 0 and injected = 1. The `obey` target SHALL use only injected conditions: 1 = `target_hit`, 0 = neither target nor repetition. Repetition-only rows SHALL be excluded and counted.

#### Scenario: Repetition excluded from obey
- **WHEN** an injected output has `repetition = true` and `target_hit = false`
- **THEN** it is not in the obey training or test set, and it is counted in `n_excluded_repetition`

#### Scenario: Low power flagged
- **WHEN** any obey class has fewer than 20 items in a modality
- **THEN** the obey rows carry `low_power = true`, and AUROC is omitted if a class has fewer than 5 items

### Requirement: Probe-evading test-only evaluation
Probes fitted on the `main` folds SHALL additionally score the `probe_evading` group for P1 and P2. Those results SHALL be reported separately.

#### Scenario: Separate rows
- **WHEN** `probe_evading` pairs exist
- **THEN** `present.csv` contains rows with `eval_group = probe_evading` for P1 and P2, and none of these pairs appears in any training fold

### Requirement: Out-of-fold scores persisted
Raw out-of-fold scores SHALL be written to disk so that the bootstrap and other re-analysis can run without refitting.

#### Scenario: Score files
- **WHEN** `probe` completes
- **THEN** `probes/scores/` contains per-probe × method × layer score files keyed by `pair_id`
