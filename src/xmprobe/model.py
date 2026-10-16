"""Model/processor loading and the batched calls the GPU stages need.

All torch/transformers imports are local so CPU-only stages and tests never need them.
Stages obtain a runner with ``get_runner(cfg)``; tests replace it with a fake.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger("xmprobe.model")
_RUNNERS: dict[str, "Runner"] = {}


class Runner:
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        mcfg = cfg["model"]
        self.device = mcfg.get("device", "cuda")  # "mps" for local Apple-silicon runs
        self.enable_thinking = mcfg.get("enable_thinking", False)
        self.processor = AutoProcessor.from_pretrained(mcfg["id"], revision=mcfg.get("revision"))
        self.processor.tokenizer.padding_side = "left"  # last position is the last prompt token
        self.model = AutoModelForImageTextToText.from_pretrained(
            mcfg["id"], revision=mcfg.get("revision"),
            torch_dtype=getattr(torch, mcfg.get("dtype", "bfloat16")), device_map=self.device,
        ).eval()
        self.torch = torch
        self.revision = getattr(self.model.config, "_commit_hash", None) or mcfg.get("revision")
        self.n_layers = self.model.config.get_text_config().num_hidden_layers
        log.info("loaded %s @ %s (%d layers)", mcfg["id"], self.revision, self.n_layers)

    def _encode(self, messages: list[list[dict]], images: list[Image.Image | None]):
        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                               enable_thinking=self.enable_thinking)
            for m in messages
        ]
        imgs = [im for im in images if im is not None]
        kwargs = {"images": imgs} if imgs else {}
        return self.processor(text=prompts, padding=True, return_tensors="pt", **kwargs).to(self.device)

    def generate(self, messages: list[list[dict]], images: list[Image.Image | None],
                 max_new_tokens: int) -> list[tuple[str, list[int]]]:
        """Greedy decoding. Returns (decoded text, generated token ids) per input."""
        inputs = self._encode(messages, images)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        new = out[:, inputs["input_ids"].shape[1]:]
        eos = {self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id}
        results = []
        for row in new.tolist():
            ids = [t for t in row if t not in eos]
            results.append((self.processor.tokenizer.decode(ids, skip_special_tokens=True), ids))
        return results

    def last_token_states(self, messages: list[list[dict]], images: list[Image.Image | None]):
        """float16 array [batch, n_layers + 1, hidden] at the last prompt token."""
        inputs = self._encode(messages, images)
        with self.torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != self.n_layers + 1:
            raise RuntimeError(f"expected {self.n_layers + 1} hidden states, got {len(hs)}")
        stacked = self.torch.stack([h[:, -1, :] for h in hs], dim=1)
        return stacked.float().cpu().numpy().astype("float16")


def get_runner(cfg: dict[str, Any]) -> Runner:
    key = f"{cfg['model']['id']}@{cfg['model'].get('revision')}"
    if key not in _RUNNERS:
        _RUNNERS[key] = Runner(cfg)
    return _RUNNERS[key]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
