# Spec Delta

## Purpose

Makes every run reproducible and resumable: configuration lives in YAML, each stage is a standalone CLI command, and every run records exactly what produced it.

## ADDED Requirements

### Requirement: Stage CLI contract
Each stage (`validate_data`, `render`, `ocr_check`, `generate`, `extract`, `probe`, `report`, `demo`, `all`) SHALL run alone as `python -m xmprobe.cli <stage> --config <yaml>`. Each SHALL accept `--limit N` and `--force`. `all` SHALL run the stages in order and stop at the first failed gate.

#### Scenario: Limit
- **WHEN** a stage runs with `--limit 10`
- **THEN** it processes only the first 10 pairs in deterministic order

#### Scenario: All stops at a gate
- **WHEN** `all` runs and `ocr_check` fails its gate
- **THEN** the image-dependent stages do not run, and the exit code is non-zero, with the gate named

### Requirement: Resumable stages
A stage SHALL skip work whose outputs already exist and validate, unless `--force` is given.

#### Scenario: Skip completed
- **WHEN** `render` runs again with an unchanged config
- **THEN** no image is rewritten and the log reports the skip

### Requirement: Configuration in YAML only
Paths, model IDs, and hyperparameters SHALL come from the YAML config. The config hash SHALL be computed over its normalized content.

#### Scenario: Config hash stable
- **WHEN** the same YAML is loaded twice with reordered keys
- **THEN** the config hash is identical

### Requirement: Run manifest
Every run SHALL write `run_manifest.json` with: the git commit, the config contents, package versions (torch, transformers, scikit-learn, pillow), the model ID and resolved revision, the GPU type, the SLURM job ID (if any), and timestamps.

#### Scenario: Manifest fields
- **WHEN** any stage runs
- **THEN** `run_manifest.json` exists and contains every listed field, with null only for the SLURM job ID outside SLURM

### Requirement: Deterministic seeds
All random choices (sampling, folds, bootstrap, email order) SHALL derive from configured seeds.

#### Scenario: Repeatable probes
- **WHEN** `probe` runs twice on the same cache
- **THEN** `present.csv` is identical across the two runs
