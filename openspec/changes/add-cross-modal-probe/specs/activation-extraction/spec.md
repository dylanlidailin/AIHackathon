# Spec Delta

## Purpose

Extracts and caches the model's internal states for every condition, so probes can be trained repeatedly on CPU without rerunning the model.

## ADDED Requirements

### Requirement: Last-prompt-token hidden states at every layer
For each condition, `extract` SHALL save the hidden state of the last prompt token (after the generation prompt) at every layer, including the embedding output. The cache shape is `[n_items, n_layers + 1, hidden_size]`.

#### Scenario: Cache shape
- **WHEN** `extract` runs on 10 pairs with a model of L layers and hidden size H
- **THEN** each `activations/{condition}.npy` has shape `[10, L+1, H]`

### Requirement: Row index
The cache SHALL include an index that maps each row to its `pair_id` and records the config hash and model revision.

#### Scenario: Index present
- **WHEN** the cache is written
- **THEN** `activations/index.json` lists the `pair_id`s in row order, together with `config_hash` and `model_revision`

### Requirement: Cache reuse
When a valid cache already exists for the same model revision, config hash, and condition, `extract` SHALL skip recomputation unless `--force` is given.

#### Scenario: Re-run skips
- **WHEN** `extract` runs a second time with an unchanged config
- **THEN** it exits without running the model and logs that the cache was reused

#### Scenario: Config change recomputes
- **WHEN** the render font size changes
- **THEN** the config hash changes and the image-condition activations are recomputed
