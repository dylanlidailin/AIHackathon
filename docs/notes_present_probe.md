# A note on the "present" probe

*Trevor, Oct 26*

One thing kept bugging me while going through the results: the "present" probe is already perfect at layer 1.

P1 (LR) is 1.00 at layer 1 in all three models, and so is P3. Even the cross-modal P2 is already 0.88 / 0.95 (LR / DoM) at layer 1 in Qwen3.5-2B and 0.93 / 0.86 in Qwen2.5-VL-3B. See `layer_sweep_present.png` in each run.

Layer 1 is one block in. The last-token state there isn't much more than a mix of the input token embeddings, so a probe that's perfect at layer 1 is mostly telling us that clean and injected documents use different words. And they do: every injected email is an LLMail attack that actually got the model to send "confirmation" to contact@contact.com, so it pretty much has to say that, while the clean side comes from a fixed pool of benign emails. A keyword search could probably do the same job.

That changes how I'd read C1. "A text-trained probe detects image injections" sounds like the model has one shared idea of "injection" across modalities. The simpler story: the model reads the image (OCR CER is 0.000 for all three), the same words end up in the residual stream, and the probe picks up the words. That's basically the critique we got about the vision encoder not being smart. It doesn't need to be. It just has to read.

Quick way to check: `scripts/lexical_baseline.py`, added with this note. It runs a keyword rule and TF-IDF + logistic regression on the raw document text, using the same team-grouped folds and AUROC as the probes. CPU only, takes seconds:

```bash
python scripts/lexical_baseline.py --config configs/qwen35_2b.yaml
```

I've only tested it on synthetic pairs since I don't have the LLMail data locally. If TF-IDF also lands near 1.0, I'd say "present" is mostly measuring wording.

If so, I think the more interesting part of our data is behavior. All three models read the images perfectly, but they act on them very differently:

| | Obeyed (text) | Obeyed (image) |
|---|---|---|
| Qwen3.5-2B | 16% | 16% |
| Qwen3.5-4B | 22% | 9% |
| Qwen2.5-VL-3B | 46% | 5% |

Qwen2.5-VL-3B is the model whose image representation looks most like text (the text threshold still works on images, accuracy 0.99), and it has the biggest gap. So reading an injection the same way doesn't mean acting on it the same way. That gap is the thing worth explaining, and the "will obey" probe gets at it more directly than "present" does.

The cheapest fix for the present probe itself: add clean emails that use the same words without being an injection, like a normal email asking the human reader to send a confirmation to contact@contact.com. If the probe still separates those from real attacks, it's picking up something beyond keywords.
