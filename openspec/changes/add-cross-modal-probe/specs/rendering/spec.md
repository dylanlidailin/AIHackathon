# Spec Delta

## Purpose

Turns pair content into images deterministically, so that the image conditions differ from the text conditions only in delivery format.

## ADDED Requirements

### Requirement: Deterministic rendering
The renderer SHALL render content as black text on a white background. It SHALL use a font file bundled in the repository and configured values for font size, width, margins, line spacing, and wrapping. Identical content and settings MUST produce byte-identical PNG files.

#### Scenario: Byte-identical output
- **WHEN** the same content is rendered twice with the same config
- **THEN** both PNG files have identical bytes

#### Scenario: Settings change output
- **WHEN** `font_size` changes from 22 to 28
- **THEN** the rendered image differs and the config hash changes

### Requirement: Both clean and injected content are rendered
For every pair the renderer SHALL produce one image of `clean_content` and one of `injected_content`, using the same settings for both.

#### Scenario: Two images per pair
- **WHEN** `render` completes
- **THEN** `images/{pair_id}_clean.png` and `images/{pair_id}_injected.png` exist for every pair

### Requirement: Oversize content is never silently truncated
If the rendered height would exceed the configured maximum, the item SHALL be dropped and logged with its `pair_id`. It MUST NOT be truncated.

#### Scenario: Oversize item
- **WHEN** content would render taller than `max_height`
- **THEN** no image is written for that pair, and the pair is listed in the render log with the reason `oversize`
