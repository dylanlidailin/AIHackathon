"""Toolchain gate (design D9 step 1): model loads, hidden states have L+1 entries, no thinking.

Usage (interactive GPU session): python scripts/check_toolchain.py configs/qwen35_0p8b.yaml
Prints the resolved revision to pin in the config (R1) and exits non-zero on failure.
"""

from __future__ import annotations

import sys

from xmprobe.config import load_config
from xmprobe.model import Runner
from xmprobe.prompts import ocr_messages
from xmprobe.render import render_text

import io
from PIL import Image


def main(path: str) -> int:
    cfg = load_config(path)
    runner = Runner(cfg)
    png = render_text("Reply with the word GOODBYE.", cfg["render"])
    img = Image.open(io.BytesIO(png)).convert("RGB")
    states = runner.last_token_states([ocr_messages()], [img])
    (text, ids), = runner.generate([ocr_messages()], [img], max_new_tokens=32)
    leaked = "<think>" in text
    ok_layers = states.shape[1] == runner.n_layers + 1
    print(f"model:          {cfg['model']['id']}")
    print(f"revision:       {runner.revision}   <- pin this in model.revision")
    print(f"hidden_states:  {states.shape[1]} (expected {runner.n_layers + 1}), hidden size {states.shape[2]}")
    print(f"output:         {text!r}")
    print(f"thinking leaked: {leaked}")
    ok = ok_layers and not leaked
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/qwen35_0p8b.yaml"))
