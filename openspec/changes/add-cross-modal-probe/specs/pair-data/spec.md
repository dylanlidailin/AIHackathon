# Spec Delta

## Purpose

Builds, validates, and splits the injection pairs that every later stage consumes, so that the text and image conditions share identical content and no item leaks across train/test splits.

## ADDED Requirements

### Requirement: Pairs file schema validation
The system SHALL load pairs from a JSONL file. Each row MUST contain `pair_id`, `source`, `system_prompt`, `user_task`, `clean_content`, `injected_content`, `injection_text`, `target_phrase`, `group` (`main` | `probe_evading`), `level`, `team_id`, and `attack_success_phi3`. A malformed row MUST be rejected, and the error MUST name the row and the field.

#### Scenario: Valid file passes
- **WHEN** `validate_data` runs on a file where every row has all required fields with correct types
- **THEN** it exits 0 and reports the row count per `group`

#### Scenario: Missing field rejected
- **WHEN** a row lacks `target_phrase`
- **THEN** `validate_data` exits non-zero and the message names the row's `pair_id` (or line number) and the missing field

### Requirement: LLMail-Inject source selection
Pairs SHALL be built only from LLMail-Inject phase-1 submissions for scenario 1 (level IDs `level1<letter>`). The main set MUST use levels a, e, and g. Levels c, d, i, and j MUST go only to the `probe_evading` group. All other levels MUST be excluded.

#### Scenario: Level routing
- **WHEN** the converter reads submissions with scenarios `level1a`, `level1c`, `level1b`, and `level2a`
- **THEN** `level1a` goes to `main`, `level1c` goes to `probe_evading`, and `level1b` and `level2a` are dropped

### Requirement: Only fully successful attacks are kept
A submission SHALL be kept only if its objectives parse and `exfil.sent`, `exfil.destination`, `exfil.content`, and `defense.undetected` are all true.

#### Scenario: Partial success dropped
- **WHEN** a submission has `exfil.sent = true` but `exfil.content = false`
- **THEN** it is excluded and counted under the reason `partial_success`

### Requirement: Attacker email length cap
An attacker email SHALL be dropped when its formatted length (subject plus body, in LLMail's email format) exceeds the configured cap (default 500 characters). This keeps it within the length range of the benign pool, so clean content can be length-matched.

#### Scenario: Long body dropped
- **WHEN** an attacker body has 900 characters and the cap is 500
- **THEN** the submission is excluded and counted under the reason `too_long`

#### Scenario: Long subject dropped
- **WHEN** an attacker body has 300 characters and its subject has 250 characters
- **THEN** the formatted email exceeds 500 characters, and the submission is excluded as `too_long`

### Requirement: Deduplication and per-team cap
Kept submissions SHALL be deduplicated by normalized body hash. At most `k` submissions per `team_id` (configurable) SHALL be sampled.

#### Scenario: Duplicate bodies collapse
- **WHEN** two submissions have bodies that differ only in whitespace
- **THEN** only one is kept

#### Scenario: Team cap enforced
- **WHEN** a team has 40 eligible submissions and `k = 5`
- **THEN** at most 5 of that team's submissions appear in the sampled pairs

### Requirement: Pair construction holds everything but the injection fixed
Each pair SHALL use the fixed scenario-1 email as Email A. `injected_content` SHALL be Email A plus the attacker email. `clean_content` SHALL be Email A plus the benign email whose length is closest to the attacker email's. The order of the two emails SHALL be randomized per pair and SHALL be the same in the clean and injected versions.

#### Scenario: Shared order and Email A
- **WHEN** a pair is built
- **THEN** Email A appears at the same position in `clean_content` and `injected_content`, and the only difference is the second email

#### Scenario: Length matching
- **WHEN** the attacker email is 300 characters long
- **THEN** the chosen benign email is the one in the pool with the smallest absolute length difference from 300

### Requirement: Reproducible sampling
Sampling SHALL be seeded (default 42). The main set SHALL contain up to 200 pairs and `probe_evading` up to 100 pairs. The converter SHALL report the number of eligible candidates per level before sampling.

#### Scenario: Same seed, same pairs
- **WHEN** the converter runs twice with the same inputs and seed
- **THEN** both runs produce byte-identical `pairs.jsonl`

#### Scenario: Insufficient candidates
- **WHEN** fewer than 200 eligible main candidates exist
- **THEN** the converter writes all of them, warns with the shortfall, and records it in the conversion report

### Requirement: Leakage-free grouped splits
Every train/test split SHALL be grouped by `team_id`. All pairs from one team, together with both modality versions of each pair, MUST fall on the same side of every split.

#### Scenario: No team on both sides
- **WHEN** any fold is generated
- **THEN** the set of `team_id`s in train and the set in test are disjoint, and this is checked by an assertion at runtime

### Requirement: Held-out probe-evading group
Pairs in the `probe_evading` group MUST NOT be used to fit any probe, select a layer, or pick a hyperparameter. They SHALL be used only as an extra test set.

#### Scenario: Excluded from training
- **WHEN** probes are fitted
- **THEN** no `probe_evading` pair is in any training fold

### Requirement: Dataset not committed
Raw LLMail files and generated `pairs.jsonl` MUST NOT be committed to the repository.

#### Scenario: Git-ignored
- **WHEN** `git status` runs after conversion
- **THEN** the files in `data/` do not appear as untracked
