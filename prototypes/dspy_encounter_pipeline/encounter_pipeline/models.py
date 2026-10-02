"""The typed shapes passed between pipeline steps.

Plain Pydantic, no DSPy import: production code (runtime.py, and later the
Frappe adapter) validates model replies against these without depending on
DSPy at all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FactType = Literal[
    "symptom",
    "history",
    "medication",
    "allergy",
    "exam_finding",
    "vital",
    "assessment",
    "plan_item",
]


class Fact(BaseModel):
    """One thing that was said in the room, with where it was said."""

    id: str = Field(default="", description="Leave empty; assigned by code after the fact is checked.")
    type: FactType
    value: str = Field(description="The finding in a few words, e.g. 'cough', 'amoxicillin 500 mg'.")
    negated: bool = Field(default=False, description="True when it was ruled out, e.g. 'no fever'.")
    detail: str = Field(default="", description="Onset, duration, severity or dose exactly as said; empty if none.")
    speaker: Literal["doctor", "patient", "companion", "unknown"]
    certainty: Literal["stated", "uncertain"] = Field(
        description="'uncertain' when the speaker hedged: 'maybe', 'I think', 'not sure'."
    )
    turns: list[str] = Field(description="Ids of the turns this fact comes from, e.g. ['t03'].")
    quote: str = Field(description="The shortest verbatim span from the cited turns that supports the fact.")


class NoteSentence(BaseModel):
    text: str
    fact_ids: list[str] = Field(description="Ids of the facts this sentence is written from, e.g. ['f2', 'f5'].")
