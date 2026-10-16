<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0 (MINOR)
- Modified: Research Constraints → Probe family (LR and difference of means are now co-primary)
- Modified: Principle I (P1 ≥ 0.80 gate applies per method)
- Modified: Principle V (splits grouped by submitting team, which subsumes pair-level grouping)
- Modified: Principle VII (bootstrap resamples the split groups)
- Modified: Compute & Environment (allocation `ai-safety-hackathon-rig`)
- Modified: Development Workflow (OpenSpec replaces Spec Kit; CLI form `python -m xmprobe.cli <stage>`)
- Artifacts reflecting this version: the local OpenSpec planning workspace (not tracked in the repo), configs/*.yaml
- Follow-up TODOs: none
-->

# Cross-Modal Injection Probes Constitution

## Core Principles

### I. Controls Before Claims (NON-NEGOTIABLE)

No cross-modal result may be reported without its controls in the same table or figure:

- **P1 text sanity probe** (train text, test held-out text) must reach AUROC ≥ 0.80 at its best layer, otherwise the probing setup itself is broken and nothing downstream is interpreted. The gate applies **per method**: a method's P2/P3 results are interpretable only if that method's P1 passes.
- **P3 image-only control probe** (train images, test held-out images) must be reported next to every **P2 cross-modal** number.
- **OCR check** must pass (see Principle II) before any image-condition probe result is interpreted as a statement about representations.

Rationale: a failed transfer is ambiguous ("no signal" vs "different signal") unless P3 and OCR disambiguate it.

### II. Only the Modality Changes

Between text and image conditions, everything except the delivery format of the content is held fixed: same model weights, same user task text, same system prompt, same content string, same decoding settings.

- The clean image condition (c) MUST also be an image, so no probe can succeed by detecting "an image is present".
- The renderer is deterministic: fixed font file shipped in the repo, fixed font size, width, margins, and wrapping.
- The OCR gate: the model must transcribe rendered clean content with median character error rate (CER) ≤ 0.10 on a sample of ≥ 30 items. If it fails, increase font size once; if it still fails, switch model rather than interpret results.

### III. Separate Metrics, Never Merged

- **Target-phrase success** (the injected target phrase appears in the output) and **degenerate repetition** (output hits the token limit in a repetition loop) are recorded as separate columns everywhere.
- "Will obey" probe labels use target-phrase success only. Repetition-only outputs are excluded from obey labels and counted separately.
- The repetition detector is the same definition as the existing text-injection pipeline, so numbers stay comparable.

### IV. Threshold-Free First

- **AUROC is the primary probe metric.** Accuracy, precision, and recall are secondary and only reported at a threshold chosen on training-modality data.
- A **per-modality centering** variant (subtract each modality's mean clean activation) is reported as a clearly labeled secondary analysis, never silently substituted for the primary result.
- Low accuracy with high AUROC is reported as "shared direction, shifted threshold", not as "transfer failed".

### V. No Leakage, No Peeking

- All splits are **grouped by submitting team** (`team_id`): all items from one team, and the text and image versions of each item, are always on the same side of any split. This subsumes grouping by pair ID and also keeps near-duplicate attack variants from one team out of the test folds.
- The headline layer for each probe is selected using **training-modality data only** (out-of-fold or inner cross-validation scores on that modality). The image data is never used to pick a layer, regularization strength, or threshold for P2.
- The full layer sweep is always shown alongside the headline number.
- If injection-template metadata exists, at least one template family is held out entirely from probe training.

### VI. Reproducible by Default

- Greedy decoding for all generations; fixed seeds for splits and probes.
- Every run writes a `run_manifest.json` with: git commit hash, config file contents, package versions (torch, transformers, scikit-learn, pillow), model ID and revision, GPU type, SLURM job ID, and timestamps.
- Configuration lives in YAML under `configs/`; no hard-coded paths or hyperparameters in code.
- Cached activations are keyed by (model revision, config hash, condition) and re-used rather than recomputed.

### VII. Scoped, Uncertain Claims

- Every claim names the model(s), renderer settings, and dataset it was measured on.
- Attack success rates carry Wilson 95% intervals; AUROCs carry group-level bootstrap 95% intervals (B = 1000), resampling the same groups used for splits.
- Text vs image attack success on the same items is compared with a paired test (McNemar).
- Small-sample results (fewer than 20 positives in any class) are flagged as low-power in tables.

### VIII. Benign and Offline

- Injected instructions use benign target phrases (e.g. a fixed nonsense token string) rather than harmful payloads.
- Experiments run only on local open-weight models on university compute. No attacks against deployed or third-party systems.
- Datasets containing real personal data are not committed to the repository.

## Research Constraints

- **Primary model:** Qwen3.5-2B. **Fast check:** Qwen3.5-0.8B. **Fallback:** Qwen2.5-VL-3B if Qwen3.5 is not supported by the pinned toolchain.
- **Scope:** image-*reading* models only; no image-generation models; no pixel-optimized attacks in this feature.
- **Probe family:** two co-primary linear methods, reported for every probe (P1, P2, P3): (a) logistic regression (LR) and (b) difference of means (DoM: direction = mean(injected) − mean(clean) on training pairs; score = projection; no fitted hyperparameters). Non-linear probes remain optional and never replace these.
- **Activation site:** the hidden state of the last prompt token (after the generation prompt) at every layer. Other sites are optional extensions.

## Compute & Environment

- Runs on UVA HPC (Rivanna/Afton) via SLURM under the `ai-safety-hackathon-rig` allocation (partition `gpu`, A100).
- The shared `torch311` conda environment is never modified; this project uses its own environment (`xmprobe`), cloned from it.
- Hugging Face cache: `/scratch/esd4uq/hf_cache`. Large artifacts (activations, generations) live under `/scratch/esd4uq/xmprobe/`; only small summaries (CSV, JSON, PNG) are committed.
- Scratch is not permanent storage: final results are copied to home or committed before the run is considered complete.
- Every stage is resumable: it skips work whose outputs already exist and validates them.

## Development Workflow

- OpenSpec flow: `proposal.md` → `specs/` + `design.md` (must respect every principle here) → `tasks.md` → implementation (`/opsx:apply`).
- Each pipeline stage is a CLI entry point (`python -m xmprobe.cli <stage> --config ...`) that can run alone.
- A smoke test (`--limit 10`, Qwen3.5-0.8B) must pass end-to-end before any full run is submitted.
- Unit tests cover: renderer determinism, data-schema validation, grouped-split leakage, repetition/target-phrase scoring.
- Results tables are regenerated from cached artifacts by a single `report` stage; numbers are never copied by hand.

## Governance

- This constitution supersedes ad-hoc decisions. Any plan that violates a principle must record the violation and its justification in the plan's Complexity Tracking table.
- Amendments are made by pull request that updates this file, bumps the version, and updates the Sync Impact Report.
- Versioning: MAJOR for removing or redefining a principle; MINOR for adding a principle or section; PATCH for wording clarifications.
- Reviewers check every results PR against Principles I, III, IV, and V before merging.

**Version**: 1.1.0 | **Ratified**: 2026-10-16 | **Last Amended**: 2026-10-16
