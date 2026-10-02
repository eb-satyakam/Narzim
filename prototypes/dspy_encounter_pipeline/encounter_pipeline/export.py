"""Turn an optimized DSPy step into plain prompt files.

DSPy is the tool that WRITES the prompt; it is not what RUNS in production.
After optimizing, this writes three files per step:

    <name>.txt            the system prompt (instruction + output format), for prompts/
    <name>.messages.json  the full message list: system, the worked examples DSPy
                          picked, and the final user turn with {placeholders}
    <name>.meta.json      version, SHA-256, eval score, model and optimizer used

The Frappe adapter then loads these like any other prompt file, so the rules
that already hold (prompts are files, versioned, hashed in AI Prompt Log,
reviewed in a PR) still hold. DSPy never becomes a production dependency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import dspy


def export_step(
    predictor: dspy.Predict,
    name: str,
    out_dir: Path,
    *,
    score: float | None = None,
    model: str = "",
    optimizer: str = "",
) -> dict[str, Path]:
    signature = predictor.signature
    placeholders = {field: "{" + field + "}" for field in signature.input_fields}
    messages = dspy.ChatAdapter().format(signature, list(predictor.demos), placeholders)

    out_dir.mkdir(parents=True, exist_ok=True)
    system = messages[0]["content"]
    paths = {
        "prompt": out_dir / f"{name}.txt",
        "messages": out_dir / f"{name}.messages.json",
        "meta": out_dir / f"{name}.meta.json",
    }
    paths["prompt"].write_text(system, encoding="utf-8")
    paths["messages"].write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["meta"].write_text(
        json.dumps(
            {
                "prompt_version": name,
                "sha256": hashlib.sha256(paths["messages"].read_bytes()).hexdigest(),
                "inputs": list(signature.input_fields),
                "outputs": list(signature.output_fields),
                "worked_examples": len(predictor.demos),
                "dev_score": score,
                "optimized_with": {"model": model, "optimizer": optimizer},
                "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
