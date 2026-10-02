"""What each pipeline step takes and returns.

A DSPy signature is the step's contract: typed inputs, typed outputs, and a
docstring that is the step's instruction. The optimizer rewrites the
instruction and picks worked examples; it never changes the types, so the code
checks in pipeline.py can rely on the shape.
"""

from __future__ import annotations

import dspy

from .models import Fact, NoteSentence


class ExtractFacts(dspy.Signature):
    """Extract every clinically relevant fact stated in this consultation.

    Only what was said: never add a finding, diagnosis or dose nobody stated.
    Keep ruled-out findings as negated facts ("no fever" is a fact).
    When a later turn corrects an earlier one, keep only the corrected version.
    Mark hedged statements as uncertain.
    Cite the turn ids each fact comes from and copy the shortest verbatim quote.
    """

    transcript: str = dspy.InputField(desc="One turn per line: '<turn id> <speaker>: <words>'.")
    facts: list[Fact] = dspy.OutputField()


class PickCatalogItem(dspy.Signature):
    """Choose the catalog entry that names the same thing as the fact.

    Answer with one candidate copied exactly, or NONE when no candidate is the
    same drug or diagnosis. A broader or more specific entry is not the same.
    """

    fact: str = dspy.InputField()
    candidates: list[str] = dspy.InputField()
    choice: str = dspy.OutputField()


class WriteSOAP(dspy.Signature):
    """Write the visit note from the facts only.

    Every sentence cites the fact ids it is written from. Never state anything
    a fact does not support, and never make a diagnosis more specific than the
    assessment fact. Negated facts are written as negatives. Uncertain facts
    keep their hedge. Recorded vitals go in Objective as given.
    """

    facts: list[Fact] = dspy.InputField()
    recorded_vitals: str = dspy.InputField(desc="Vitals already on the record, or 'none'.")
    language: str = dspy.InputField(desc="Language to write the note in, e.g. 'en-US', 'es-MX'.")
    subjective: list[NoteSentence] = dspy.OutputField()
    objective: list[NoteSentence] = dspy.OutputField()
    assessment: list[NoteSentence] = dspy.OutputField()
    plan: list[NoteSentence] = dspy.OutputField()
