"""Load the answer-key cases into DSPy examples."""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from .pipeline import Turn

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "data" / "fact_cases.jsonl"


def load_cases(path: Path = DEFAULT_CASES) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """(train, dev). Train is what the optimizer learns from; dev is only scored."""
    train: list[dspy.Example] = []
    dev: list[dspy.Example] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        example = dspy.Example(
            name=case["name"],
            turns=[Turn(*t) for t in case["turns"]],
            gold_facts=case["gold_facts"],
        ).with_inputs("turns")
        (dev if case["split"] == "dev" else train).append(example)
    return train, dev
