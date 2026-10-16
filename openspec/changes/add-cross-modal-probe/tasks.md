# Tasks

## 1. Repository and project rules

- [ ] 1.1 Commit the current Spec Kit docs and the zip once (a snapshot for rollback). Verify with `git log -1 --stat`, which lists them.
- [ ] 1.2 Rename `README-2.md` → `README.md`, replace the Spec Kit links (`specs/001-…`, `.specify/…`) with `openspec/` paths, and add the LLMail citation. Verify that `grep -n "specs/001\|.specify" README.md` returns nothing.
- [ ] 1.3 Move `quickstart.md` → `docs/quickstart.md` and the PDF → `docs/papers/`. Delete `plan.md` and `xmodal-injection-probe.zip`. Verify with `ls`, which shows only the README, the constitution, docs/, and openspec/ at the root.
- [ ] 1.4 Amend `constitution.md` to v1.1.0 per design D10: LR + DoM co-primary, P1 gate per method, grouping by `team_id`, account `ai-safety-hackathon-rig`, updated Sync Impact Report and dates. Verify that the version line reads 1.1.0 and that no `aikyamlab` remains.
- [ ] 1.5 Add `context` and `rules` to `openspec/config.yaml` per D10. Verify that `openspec validate add-cross-modal-probe` still passes.
- [ ] 1.6 Add a `.gitignore` covering `data/`, `logs/`, `__pycache__/`, and `*.npy`. Verify that `git check-ignore data/pairs.jsonl` succeeds.

## 2. Package scaffold, config, and provenance

- [ ] 2.1 Create `pyproject.toml` (package `xmprobe`, `src/` layout, pytest config) and an empty module skeleton per D1. Verify that `pip install -e .` and `python -m xmprobe.cli --help` work.
- [ ] 2.2 Implement `config.py`: YAML load and a normalized config hash. Write `configs/qwen35_2b.yaml`, `qwen35_0p8b.yaml`, and `qwen25vl_3b.yaml` with the D7 fields. Verify that a test shows reordered keys give the same hash.
- [ ] 2.3 Implement the `cli.py` dispatcher, with `--limit`/`--force`, skip-if-valid handling, `run_manifest.json`, and stop-at-failed-gate for `all`. Verify that tests cover the manifest fields and the skip behavior (run-provenance spec).
- [ ] 2.4 Bundle `assets/fonts/DejaVuSans.ttf` with its license file. Verify that the file exists and that its hash is recorded in the manifest.

## 3. Pair data (LLMail)

- [ ] 3.1 Write `scripts/convert_llmail.py`: stream the raw phase-1 JSONL, route by level, apply the full-success filter, the 450-character cap, dedupe, and the per-team cap, then write a conversion report with counts per level and per drop reason. Verify with unit tests on a fixture of about 20 synthetic rows covering every drop reason.
- [ ] 3.2 Implement pair construction: fixed scenario-1 Email A, a length-matched benign email, a seeded shared order, and the LLMail email format. Verify with tests showing that clean and injected differ only in the second email, and that the same seed gives identical output.
- [ ] 3.3 Implement the `data.py` schema validation and the `validate_data` stage. Verify with accept/reject tests (pair-data spec scenarios).
- [ ] 3.4 Implement the grouped fold generator (by `team_id`, seed 42), excluding `probe_evading`. Verify that `tests/test_splits.py` asserts disjoint teams and no evading pairs in train.
- [ ] 3.5 On Rivanna, download the LLMail files (design D2 command) into `/scratch/esd4uq/hf_cache`, run the converter, and record the candidate counts. Verify that the conversion report exists. If main < 200, apply the D2/Risks fallback and note it.

## 4. Rendering and prompts

- [ ] 4.1 Implement `render.py` (D5) and the `render` stage, including dropping and logging oversize items. Verify that `tests/test_render.py` shows byte-identical PNGs for identical input and different PNGs when the font size changes.
- [ ] 4.2 Implement `prompts.py`: the LLMail system prompt with `tool_prompt`, the user task, and the `<document>` slot with text or an image. Verify with a test that the four prompts are equal once the content slot is masked.

## 5. Model, toolchain, and environment (Rivanna)

- [ ] 5.1 Create the `xmprobe` env by cloning `torch311` and install the dependencies (docs/quickstart.md). Verify that the printed torch and transformers versions are recorded.
- [ ] 5.2 Implement `model.py`: load the model and processor, use bf16, and pin and resolve the revision. Write `scripts/check_toolchain.py`.
- [ ] 5.3 In an interactive `srun` session (D8), run the toolchain check on 0.8B and 2B. Verify that `hidden_states == L+1`, that no `<think>` appears, and that R1/R3 are recorded in the configs. If this fails for more than 15 minutes, switch to `qwen25vl_3b.yaml`.

## 6. OCR gate, generation, and scoring

- [ ] 6.1 Implement `ocr_check.py`: seeded sample, transcription, CER, and a summary with `pass`/`reason`. Verify with unit tests on CER and the `n < 30` rule.
- [ ] 6.2 Implement `scoring.py`: `target_hit`, `destination_hit`, `content_hit` (LLMail JSON and native tool-call markup), repetition, Wilson CIs, and McNemar. Verify that `tests/test_scoring.py` covers the spec scenarios (full hit, destination only, prose mention, looping).
- [ ] 6.3 Implement `generate.py` (greedy, batched, thinking off) and `behavior_summary.csv` per group. Verify on a 3-pair run that the CSV has all columns.

## 7. Activation extraction

- [ ] 7.1 Implement `extract.py`: last prompt token, all layers, float16 cache, `index.json`, and reuse keyed by the config hash. Verify on 10 pairs that the shape is `[10, L+1, H]` and that a second run skips the work.

## 8. Probes

- [ ] 8.1 Implement LR and DoM for P1/P2/P3 with grouped outer CV, an inner CV for C (with `n_jobs`), pooled out-of-fold scores, and a group bootstrap. Verify with a synthetic-activation test where an injected signal is planted along a known direction: P1, P2, and P3 AUROC are all > 0.95.
- [ ] 8.2 Add `tests/test_dom_no_leakage.py`. Verify that each fold's DoM direction uses only that fold's training `team_id`s.
- [ ] 8.3 Add per-method headline-layer selection from training-modality data only, the centered variant (accuracy only), `probe_evading` test-only scoring, and score persistence. Verify with tests that P2 layer selection never reads image arrays.
- [ ] 8.4 Add obey targets: exclusion of repetition-only rows, class counts, the low-power flag, and AUROC omitted below 5 items. Verify with a unit test on a crafted label set.

## 9. Report and demo

- [ ] 9.1 Implement `report.py`: sweep plots (LR solid, DoM dashed), `headline.md` with the OCR and P1 per-method gates, the method-agreement check, outcome classification, length gap, scope line, and low-power flags. Verify with a test that a fixture with LR P1 = 0.92 and DoM P1 = 0.71 labels DoM "not interpretable (P1)", and that a second run is byte-identical.
- [ ] 9.2 Implement the `demo` stage: a stratified 100-pair sample, mini OCR, extract, present probes, and a report into `demo/`, labeled "signal check" and without classification. Verify with a 10-pair demo run on 0.8B that the labels are present.
- [ ] 9.3 Write `slurm/gpu_stage.sbatch` and `slurm/cpu_probe.sbatch` with account `ai-safety-hackathon-rig` (D8). Update `docs/quickstart.md` with the `srun` line, the demo step 3b, and the LLMail download. Verify that `sbatch --test-only slurm/gpu_stage.sbatch configs/qwen35_0p8b.yaml render` is accepted. Get user approval before any real submission.

## 10. Runs (integration)

- [ ] 10.1 Smoke run: `all --limit 10` on 0.8B in an interactive session. Verify that every stage produces valid output in < 20 min.
- [ ] 10.2 Demo run on 2B with 100 pairs. Verify that `demo/reports/headline.md` exists and that the manifest wall time is < 15 min (SC-007).
- [ ] 10.3 OCR gate on 2B (n = 40). Verify that `ocr_summary.json` has `"pass": true`, or apply the D5 fallback chain.
- [ ] 10.4 Full run: chained `generate` → `extract` → `probe` + `report` (after user approval of the `sbatch` submission). Verify that `headline.md` contains a classified outcome or an explicit gate label.
- [ ] 10.5 Copy the summaries to `results/<run_name>/` and commit. Verify with `git show --stat`, which lists `headline.md`, the PNGs, the CSVs, and the manifest.

## Workflow follow-up

- Archive the change with `/opsx-archive` after the results are committed and reviewed against Constitution Principles I, III, IV, and V.
