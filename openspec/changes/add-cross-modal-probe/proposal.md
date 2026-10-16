# Proposal

## Why

Prompt-injection detectors are trained and evaluated almost entirely on text, while agents increasingly read images (screenshots, rendered pages, scans). Linear probes on VLM activations detect injections within one modality (ARGUS), but nobody has tested whether a probe trained on **text** detects the **same** injections delivered as **rendered images** in the same model. This change builds the pipeline that answers that question for the AI Safety Hackathon, with controls that make a negative result interpretable.

## What Changes

- New research pipeline (`xmprobe`) run as resumable CLI stages on UVA HPC (Rivanna), allocation `ai-safety-hackathon-rig`, one A100.
- Dataset: pairs built from Microsoft LLMail-Inject (phase 1, scenario 1). A `main` set (levels a/e/g) and a test-only `probe_evading` set (levels c/d/i/j).
- Four input conditions per pair (clean/injected × text/image) that differ only in how the content is delivered and in whether the injection is present.
- Deterministic rendering, an OCR readability gate, greedy generation, and separate target-phrase and repetition scoring.
- Last-prompt-token activations at every layer, cached on scratch.
- "Injection present" probes P1 (text→text), P2 (text→image), and P3 (image→image). Each is computed with two co-primary linear methods: logistic regression (LR) and difference of means (DoM).
- "Will obey" probes on the same grid. These are best-effort and flagged as low-power when underpowered.
- One-command report with an outcome classification, a method-agreement check, and a length-confound check.
- A fast `demo` stage: 100 pairs, present probes only, under 15 minutes on Qwen3.5-2B.
- Constitution amended to v1.1.0 (LR + DoM co-primary, P1 gate per method). Its key gates are also mirrored into `openspec/config.yaml` rules.
- Repository tidy-up: README-2 becomes the README, the quickstart and the reference paper move to `docs/`, and the duplicate Spec Kit zip is removed.

**Delivery priority (hackathon):** must have: behavioral baseline + OCR gate, extraction, present probes, report, demo. Best effort: obey probes. Not in this change: activation steering (spec User Story 6).

## Capabilities

### New Capabilities

- `pair-data`: building, validating, and splitting LLMail-derived injection pairs (schema, sampling, grouping, held-out sets).
- `rendering`: deterministic text-to-image rendering of pair content.
- `ocr-gate`: checking that the model can read rendered content before any image result is interpreted.
- `generation-scoring`: greedy generation in four conditions and behavioral metrics (target hit, repetition, intervals, paired test).
- `activation-extraction`: cached last-prompt-token hidden states per condition and layer.
- `probing`: grouped-CV linear probes (LR + DoM) for present and obey targets, P1/P2/P3, with uncertainty.
- `reporting`: regenerating figures and tables from cached artifacts, interpretation gates, and outcome classification.
- `run-provenance`: per-run manifest, YAML configuration, resumable stages, and the CLI contract.
- `demo-run`: a fast sampled end-to-end run that is informational only.

### Modified Capabilities

None (no existing specs).

## Impact

- New code: `src/xmprobe/`, `scripts/convert_llmail.py`, `configs/*.yaml`, `slurm/*.sbatch`, `tests/`, and `assets/fonts/DejaVuSans.ttf`.
- New external dependencies: Hugging Face models (`Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-0.8B`, fallback `Qwen/Qwen2.5-VL-3B-Instruct`), the `microsoft/llmail-inject-challenge` dataset (MIT, ~1.7 GB raw phase-1 file), and a project conda env `xmprobe` cloned from `torch311`.
- Compute: Rivanna `gpu` partition, account `ai-safety-hackathon-rig`, A100, 16 CPU / 128 GB / 4 h. Storage goes under `/scratch/esd4uq/`.
- Docs: `constitution.md` goes to v1.1.0, and `openspec/config.yaml` gains context and rules. The README, quickstart, and plan are reorganized, and Spec Kit paths (`.specify/`, `specs/001-…`) are retired in favor of OpenSpec.
