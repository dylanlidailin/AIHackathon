# Spec Delta

## Purpose

Regenerates every figure and table from cached artifacts and applies the interpretation gates, so that no headline is reported without its controls.

## ADDED Requirements

### Requirement: Reproducible report from cache
The `report` stage SHALL build all figures and tables from cached artifacts only, without running the model. Deleting `reports/` and re-running SHALL reproduce identical tables.

#### Scenario: Regeneration
- **WHEN** `reports/` is deleted and `report` runs again
- **THEN** the CSV and Markdown outputs are byte-identical to the previous run

### Requirement: Layer sweep figures
The report SHALL plot the full layer sweep of AUROC with CIs for P1, P2, and P3, for each target. LR is drawn solid and DoM dashed.

#### Scenario: Sweep plot
- **WHEN** present-probe results exist
- **THEN** `reports/layer_sweep_present.png` shows P1/P2/P3 × {LR, DoM} across all layers

### Requirement: P3 shown beside every P2
Every table or figure that shows a P2 number SHALL also show the matching P3 number.

#### Scenario: Headline table
- **WHEN** `headline.md` reports P2 for a method
- **THEN** the same row or table shows that method's P3

### Requirement: Per-method P1 gate
If a method's P1 AUROC at its headline layer is below 0.80, that method's P2/P3 results SHALL be labeled "not interpretable (P1)".

#### Scenario: One method fails P1
- **WHEN** LR P1 = 0.92 and DoM P1 = 0.71
- **THEN** DoM P2/P3 are labeled "not interpretable (P1)", and only LR is used for classification

### Requirement: Method-agreement check
If both methods pass P1 and their P2 AUROC 95% CIs do not overlap, the result SHALL be labeled "method-dependent" and no outcome SHALL be classified.

#### Scenario: Disagreement
- **WHEN** LR P2 CI = [0.85, 0.95] and DoM P2 CI = [0.55, 0.70]
- **THEN** `headline.md` states "method-dependent" and shows both headline layers

### Requirement: Outcome classification
When the gates pass, the report SHALL classify the result into exactly one row of the interpretation table: P2 works; P2 fails and P3 works; both fail with OCR passing; high AUROC with low accuracy ("shared direction, shifted threshold"). The table and the controls that justify the classification SHALL be quoted next to it.

#### Scenario: Classified outcome
- **WHEN** OCR passes, P1 passes, the methods agree, P2 AUROC is 0.55 [0.45, 0.65], and P3 is 0.90
- **THEN** `headline.md` classifies the result as "P2 fails, P3 works"

### Requirement: Confound and scope disclosure
The headline SHALL report the mean and max character-length gap between `clean_content` and `injected_content`. It SHALL name the model, the renderer settings, and the dataset. Groups with fewer than 20 positives in any class SHALL be flagged as low-power.

#### Scenario: Length gap shown
- **WHEN** `headline.md` is written
- **THEN** it contains a "Length gap" line with the mean and max over the pairs that were evaluated

### Requirement: Results copied out of scratch
Summary outputs (reports, CSVs, manifest) SHALL be copyable to `results/<run_name>/` in the repository, with one documented command.

#### Scenario: Results committed
- **WHEN** the documented copy step runs
- **THEN** `results/<run_name>/` contains `headline.md`, the sweep PNGs, the summary CSVs, and `run_manifest.json`
