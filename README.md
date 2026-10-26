# Cross-Modal Injection Probes

> Do text-trained prompt-injection probes detect the same injections when they arrive as images?

Status: runs complete on Qwen3.5-2B, Qwen3.5-4B and Qwen2.5-VL-3B (173 + 58 LLMail pairs each). HPC workflow in [`docs/quickstart.md`](docs/quickstart.md), project rules in [`constitution.md`](constitution.md).

## TL;DR

Prompt-injection defenses are mostly text-based, but agents increasingly read images: screenshots, scanned documents, rendered web pages. We train a linear probe on a vision-language model's activations to detect injections using text inputs only, then test it, without retraining, on the same injections rendered as images.

It transfers: in all three models the text-trained probe separates clean from injected images almost perfectly (AUROC ≥ 0.998). We now suspect this mostly shows the model *reading* the attack wording, and that the more interesting finding is in behavior. See [What we think this means](#what-we-think-this-means).

## Background

- **Indirect prompt injection.** An attacker hides instructions in content an agent reads (an email, a web page, a tool output) so the agent follows them instead of the user. The injected task can be harmless; the problem is that the user never asked for it.
- **The modality gap** (Zhang et al., 2026). VLMs follow an instruction given as text far more often than the same instruction shown as an image of text (2× to 300×). Their defense, rendering untrusted text as an image, does nothing when the content is already an image.
- **Within-modality probes** (ARGUS, Lu et al., 2025). Linear probes on VLM activations separate "following the user" from "following the injection" almost perfectly, but were trained and tested within one modality. Whether a signal learned from text transfers to images hadn't been tested.

## Research questions

| ID | Question |
|---|---|
| RQ1 | Does our injection set reproduce the modality gap (text attacks succeed more often than image attacks)? |
| RQ2 | Does an "injection present" probe trained on text detect image-delivered injections? |
| RQ3 | Does a "will obey" probe trained on text predict whether the model obeys image-delivered injections? |
| RQ4 | If the model registers image injections but rarely acts on them, is that an internal account of the modality gap? |

## Design

The user's task is always plain text. Only the document the model reads changes:

| | Document as text | Document rendered as an image |
|---|---|---|
| **Clean** | (a) | (c) |
| **Injected** | (b) | (d) |

The clean baseline on the image side (c) is also an image, so a probe can't succeed just by detecting that an image is present.

Three probes per layer, each fit with logistic regression (LR) and difference of means (DoM) on the last prompt token's hidden state:

| Probe | Train on | Test on | Role |
|---|---|---|---|
| P1 | text (a vs b) | held-out text | Confirms the signal exists |
| P2 | text (a vs b) | images (c vs d) | **Main question** |
| P3 | images (c vs d) | held-out images | Control for P2 |

Controls: an OCR check (the model must transcribe the rendered images before image results count), AUROC as the main metric (a probe can point the right way but have the wrong threshold on a new modality), splits grouped by submitting team, the headline layer chosen on text data only, and "target phrase emitted" reported separately from degenerate repetition.

Models: Qwen3.5-2B (primary), Qwen3.5-4B (scale check), Qwen2.5-VL-3B (different architecture and vision encoder), Qwen3.5-0.8B (smoke test only).

## Results

Criteria C1–C5 were fixed before the 4B and 2.5-VL-3B runs. Full numbers: [`results/cross_model/comparison.md`](results/cross_model/comparison.md).

| Criterion | Qwen3.5-2B | Qwen3.5-4B | Qwen2.5-VL-3B |
|---|---|---|---|
| **C1** late-layer text→image transfer (P2 AUROC) | ✓ 1.000 / 0.999 | ✓ 1.000 / 1.000 | ✓ 1.000 / 0.998 |
| C2 threshold shift (P2 accuracy at the text threshold) | ✓ 0.50 / 0.50 | ✓ 0.50 / 0.64 | ✗ 0.99 / 0.91 |
| C3 mid-depth dip | ✓ | ✗ | ✗ |
| C4 text/image directions align only late | ✓ | ✓ | ✗ |
| C5 "will obey" does not transfer | ✓ | ✗ | ✗ (low power) |
| Obeyed injections, text vs image | 16% vs 16% | 22% vs 9% | 46% vs 5% |

Values are LR / DoM. Only C1 holds in all three models. Scope: these models, a 22 px DejaVu Sans renderer, LLMail-Inject scenario 1.

## What we think this means

The "present" result is less impressive than it looks. The probes are already near perfect at layer 1, which is barely past the token embeddings: P1 and P3 are 1.00 there in all three models, and P2 is already 0.86–0.95 in Qwen3.5-2B and Qwen2.5-VL-3B. The injected emails are LLMail attacks that got the model to send "confirmation" to contact@contact.com, so they say so, while the clean emails come from a fixed pool of benign ones. The probe is probably picking up wording. The model reads the image perfectly (OCR error rate 0 in all three), the same words end up in its hidden states, and a text-trained probe finds them. The vision encoder doesn't have to be smart for this to work. It just has to read.

The behavior row is more interesting. All three models read the images perfectly but act on them very differently. Qwen2.5-VL-3B represents image injections most like text (the text threshold still works on images, C2) and yet has the biggest gap, 46% vs 5%. Reading an injection the same way doesn't mean acting on it the same way, so the modality gap seems to come from the language model's decision to act, not from what it can see.

More detail in [`docs/notes_present_probe.md`](docs/notes_present_probe.md).

## Next steps

1. **Run the lexical baseline.** [`scripts/lexical_baseline.py`](scripts/lexical_baseline.py) classifies the raw document text with a keyword rule and TF-IDF, using the same folds as the probes (CPU, seconds). If it also reaches ~1.0, "present" is measuring wording.
2. **Add negatives with the same wording.** For example, a normal email asking the human reader to send a confirmation to contact@contact.com. If the probe still separates these from real attacks, it is tracking more than keywords.
3. **Make "will obey" the headline.** Predicting whether the model will act is what a defense needs. Image positives are scarce right now (9 to 28 per model), so this needs more data, e.g. LLMail scenario 2.

## Running

See [`docs/quickstart.md`](docs/quickstart.md) for the HPC workflow: environment, model download, smoke test, full run, collecting results.

## Repository layout

```
.
├── constitution.md      # project rules
├── docs/                # quickstart, notes, reference papers
├── configs/             # one YAML per model
├── src/xmprobe/         # pipeline code (CLI stages)
├── scripts/             # data conversion, toolchain check, post-hoc analyses
├── slurm/               # job scripts
├── results/             # committed summaries per run
└── tests/
```

## Related work

- Abdelnabi et al. *LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge* (2025). [arXiv:2506.09956](https://arxiv.org/abs/2506.09956). Source of the injection pairs (`microsoft/llmail-inject-challenge`, MIT).
- Zhang, Baroian, van Rijn, Shafran, Tramèr. *Render Before Reading: Visual Rendering as a Prompt Injection Defense* (2026). [arXiv:2609.36121](https://arxiv.org/abs/2609.36121)
- Lu et al. *ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior* (2025). [arXiv:2512.05745](https://arxiv.org/abs/2512.05745)
- Özdinçer, Simko, Schölkopf, Jin. *Improved Robustness against Indirect Prompt Injection Can Be Built into LLMs* (COLM 2026). [GitHub](https://github.com/memo-ozdincer/odile)
- Cao et al. *VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents* (ICLR 2026). [arXiv:2506.02456](https://arxiv.org/abs/2506.02456)
- Du et al. *SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents* (2026). [arXiv:2604.25562](https://arxiv.org/abs/2604.25562)

## Ethics and scope

This project studies detection to support defenses. All experiments run offline on local open-weight models with benign target phrases, and nothing is run against deployed or third-party systems. Image-generation models and pixel-optimized adversarial images are out of scope.
