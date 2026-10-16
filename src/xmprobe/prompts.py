"""Builds the four input conditions (design D4) as chat messages.

Only the document slot differs between conditions: inline text for ``text_*`` and a
rendered image for ``image_*``. System prompt (LLMail email_client + tool_prompt), user
task and surrounding wording are identical.
"""

from __future__ import annotations

from typing import Any

CONDITIONS = ("text_clean", "text_injected", "image_clean", "image_injected")
DOC_OPEN, DOC_CLOSE = "\n\n<document>\n", "\n</document>"
OCR_INSTRUCTION = "Transcribe all text in the image exactly, preserving line breaks. Output only the text."


def split_condition(condition: str) -> tuple[str, str]:
    """``image_clean`` -> ("image", "clean")."""
    modality, kind = condition.split("_", 1)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    return modality, kind


def build_messages(pair: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    """Chat messages in the processor's list-of-parts format. Image conditions contain one
    ``{"type": "image"}`` part; the caller passes the rendered PIL image alongside."""
    modality, kind = split_condition(condition)
    messages: list[dict[str, Any]] = []
    if pair.get("system_prompt"):
        messages.append({"role": "system", "content": [{"type": "text", "text": pair["system_prompt"]}]})
    if modality == "text":
        content = pair[f"{kind}_content"]
        parts = [{"type": "text", "text": pair["user_task"] + DOC_OPEN + content + DOC_CLOSE}]
    else:
        parts = [
            {"type": "text", "text": pair["user_task"] + DOC_OPEN},
            {"type": "image"},
            {"type": "text", "text": DOC_CLOSE},
        ]
    messages.append({"role": "user", "content": parts})
    return messages


def masked(messages: list[dict[str, Any]], pair: dict[str, Any]) -> str:
    """Flatten messages with the document slot replaced by ``<SLOT>`` (for equality tests)."""
    out = []
    for m in messages:
        text = ""
        for part in m["content"]:
            if part["type"] == "image":
                text += "<SLOT>"
            else:
                t = part["text"]
                for key in ("clean_content", "injected_content"):
                    t = t.replace(DOC_OPEN + pair[key] + DOC_CLOSE, DOC_OPEN + "<SLOT>" + DOC_CLOSE)
                text += t
        out.append(f"{m['role']}:{text}")
    return "\n".join(out)


def ocr_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_INSTRUCTION}]}]
