"""Deterministic text-to-image renderer and the ``render`` stage (design D5).

Black text on white, bundled font, pixel-width word wrap. Paragraph breaks in the
content are kept. Words wider than a line are split by character. Items taller than
``max_height`` are dropped and logged, never truncated.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .common import is_done, mark_done, stamp_key, write_jsonl
from .config import run_dir
from .data import load_pairs

log = logging.getLogger("xmprobe.render")
SECTIONS = ("render",)


class Oversize(ValueError):
    pass


def _font(rcfg: dict[str, Any]) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(rcfg["font_path"], rcfg["font_size"])


def wrap(text: str, font: ImageFont.FreeTypeFont, max_px: float) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split(" ")
        line = ""
        for word in words:
            while font.getlength(word) > max_px:  # hard-split very long tokens
                cut = len(word)
                while cut > 1 and font.getlength(word[:cut]) > max_px:
                    cut -= 1
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{line} {word}" if line else word
            if font.getlength(candidate) <= max_px:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def render_text(text: str, rcfg: dict[str, Any]) -> bytes:
    """Return PNG bytes. Raises Oversize if the image would exceed ``max_height``."""
    font = _font(rcfg)
    width, margin = rcfg["width"], rcfg["margin"]
    lines = wrap(text, font, width - 2 * margin)
    line_h = round(rcfg["font_size"] * rcfg["line_spacing"])
    height = 2 * margin + line_h * len(lines)
    if height > rcfg["max_height"]:
        raise Oversize(f"height {height} > max_height {rcfg['max_height']}")
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((margin, margin + i * line_h), line, font=font, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def image_path(out: Path, pair_id: str, kind: str) -> Path:
    return out / "images" / f"{pair_id}_{kind}.png"


def render_pairs(pairs: list[dict[str, Any]], out: Path, rcfg: dict[str, Any]) -> tuple[list[str], list[dict]]:
    """Render both images per pair under ``out/images``; log oversize drops to ``out/render_log.jsonl``."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    written, dropped = [], []
    for p in pairs:
        try:
            pngs = {kind: render_text(p[f"{kind}_content"], rcfg) for kind in ("clean", "injected")}
        except Oversize as e:
            dropped.append({"pair_id": p["pair_id"], "reason": "oversize", "detail": str(e)})
            for kind in ("clean", "injected"):  # stale images from an earlier run
                image_path(out, p["pair_id"], kind).unlink(missing_ok=True)
            continue
        for kind, data in pngs.items():
            path = image_path(out, p["pair_id"], kind)
            path.write_bytes(data)
            written.append(str(path.relative_to(out)))
    write_jsonl(out / "render_log.jsonl", dropped)
    return written, dropped


def run(cfg: dict[str, Any], limit: int | None = None, force: bool = False) -> dict[str, Any]:
    out = run_dir(cfg)
    key = stamp_key(cfg, SECTIONS, limit)
    if not force and is_done(out, "render", key):
        log.info("render: outputs valid, skipping")
        return {"skipped": True}
    written, dropped = render_pairs(load_pairs(cfg, limit), out, cfg["render"])
    mark_done(out, "render", key, written + ["render_log.jsonl"])
    log.info("render: %d images, %d pairs dropped", len(written), len(dropped))
    return {"n_images": len(written), "n_dropped": len(dropped)}


def rendered_pair_ids(cfg: dict[str, Any]) -> set[str]:
    """Pairs that have both images (oversize pairs are excluded downstream)."""
    out = run_dir(cfg) / "images"
    clean = {p.name[: -len("_clean.png")] for p in out.glob("*_clean.png")}
    injected = {p.name[: -len("_injected.png")] for p in out.glob("*_injected.png")}
    return clean & injected
