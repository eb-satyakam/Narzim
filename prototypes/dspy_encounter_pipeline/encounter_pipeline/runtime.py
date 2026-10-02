"""Production side: run an exported step with no DSPy installed.

This is roughly what the Frappe adapter would do: load the exported messages,
fill the placeholders, send them through the existing provider client
(OpenAI or Anthropic, chosen by AI Settings), and validate the reply against
the Pydantic model. Nothing here imports dspy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import TypeAdapter

from .models import Fact

_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\s*(.*?)(?=\[\[ ## \w+ ## \]\]|\Z)", re.DOTALL)


def render(messages_path: Path, **inputs: str) -> list[dict[str, str]]:
    """The exported message list with every {placeholder} filled.

    Only the last message carries placeholders; the worked examples are fixed
    text. A missing input raises rather than sending a literal '{transcript}'.
    """
    messages = json.loads(Path(messages_path).read_text(encoding="utf-8"))
    last = messages[-1]["content"]
    if missing := set(re.findall(r"\{(\w+)\}", last)) - set(inputs):
        raise ValueError(f"inputs not supplied: {sorted(missing)}")
    for key, value in inputs.items():
        last = last.replace("{" + key + "}", value)
    return [*messages[:-1], {**messages[-1], "content": last}]


def parse_fields(reply: str) -> dict[str, str]:
    """Split a reply into its '[[ ## field ## ]]' sections."""
    return {name: body.strip() for name, body in _FIELD.findall(reply) if name != "completed"}


def parse_facts(reply: str) -> list[Fact]:
    """The facts in a reply, validated. Raises on a malformed reply (log it as Failed, BE-7)."""
    raw = parse_fields(reply)["facts"]
    return TypeAdapter(list[Fact]).validate_python(json.loads(raw))
