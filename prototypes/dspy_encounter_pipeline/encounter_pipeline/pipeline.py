"""The encounter pipeline: facts -> catalog match -> SOAP, with code checks between.

Only step 1 reads the transcript. Every AI step is followed by a check in plain
Python, so a model mistake is caught at the step that made it instead of
surfacing three steps later in the note.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

import dspy

from .models import Fact, NoteSentence
from .signatures import ExtractFacts, PickCatalogItem, WriteSOAP

# Which catalog a fact type is matched against. Types not listed are not coded.
CATALOG_FOR_TYPE = {"medication": "medications", "assessment": "diagnoses"}
MAX_CANDIDATES = 5
SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")


@dataclass
class Turn:
    id: str
    speaker: str
    text: str


def format_transcript(turns: list[Turn]) -> str:
    return "\n".join(f"{t.id} {t.speaker}: {t.text}" for t in turns)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ── Step 1 check: every fact must point at real words ────────────────────────


def ground_facts(facts: list[Fact], turns: list[Turn]) -> tuple[list[Fact], list[Fact]]:
    """Keep facts whose cited turns exist and contain the quote; reject the rest.

    This is the hallucination gate. A fact the model invented has no turn that
    says it, so it cannot pass, whatever the prompt said.
    """
    by_id = {t.id: _norm(t.text) for t in turns}
    kept: list[Fact] = []
    rejected: list[Fact] = []
    for fact in facts:
        cited = [by_id[t] for t in fact.turns if t in by_id]
        if fact.turns and len(cited) == len(fact.turns) and _norm(fact.quote) in " ".join(cited):
            kept.append(fact.model_copy(update={"id": f"f{len(kept) + 1}"}))
        else:
            rejected.append(fact)
    return kept, rejected


# ── Step 2: code finds candidates, a small model picks one ───────────────────


def catalog_candidates(value: str, catalog: list[str], k: int = MAX_CANDIDATES) -> list[str]:
    """Top-k catalog entries for a fact, by substring then fuzzy match.

    Replaces pasting the whole catalog into the prompt: the model only ever
    sees k entries, however large the clinic's catalog grows.
    """
    target = _norm(value)
    normed = {entry: _norm(entry) for entry in catalog}
    substring = [e for e, n in normed.items() if target and (target in n or n in target)]
    fuzzy = difflib.get_close_matches(target, list(normed.values()), n=k, cutoff=0.6)
    fuzzy_entries = [e for e, n in normed.items() if n in fuzzy]
    seen: dict[str, None] = dict.fromkeys(substring + fuzzy_entries)
    return list(seen)[:k]


# ── Step 3 check: every note sentence must cite a fact ───────────────────────


def check_note(note: dict[str, list[NoteSentence]], facts: list[Fact]) -> tuple[list[str], list[str]]:
    """(uncited sentences, facts the note never used).

    An uncited sentence is the note saying something no fact supports. A fact
    the note never used is a gap: said in the room, missing from the record.
    """
    ids = {f.id for f in facts}
    uncited: list[str] = []
    used: set[str] = set()
    for section in SOAP_SECTIONS:
        for sentence in note.get(section, []):
            valid = [i for i in sentence.fact_ids if i in ids]
            if not valid:
                uncited.append(f"{section}: {sentence.text}")
            used.update(valid)
    missing = [f.id for f in facts if f.id not in used]
    return uncited, missing


# ── The modules DSPy optimizes ───────────────────────────────────────────────


class FactExtractor(dspy.Module):
    """Step 1 plus its check. This is the unit the optimizer tunes on its own."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(ExtractFacts)

    def forward(self, turns: list[Turn]) -> dspy.Prediction:
        raw = self.extract(transcript=format_transcript(turns)).facts
        facts, rejected = ground_facts(raw, turns)
        return dspy.Prediction(facts=facts, rejected=rejected)


class EncounterPipeline(dspy.Module):
    def __init__(self, catalogs: dict[str, list[str]]) -> None:
        super().__init__()
        self.catalogs = catalogs
        self.facts = FactExtractor()
        self.pick = dspy.Predict(PickCatalogItem)
        self.write = dspy.Predict(WriteSOAP)

    def match(self, fact: Fact) -> str | None:
        catalog = self.catalogs.get(CATALOG_FOR_TYPE.get(fact.type, ""), [])
        if fact.negated or not catalog:
            return None
        candidates = catalog_candidates(f"{fact.value} {fact.detail}".strip(), catalog)
        if not candidates:
            return None
        choice = self.pick(fact=f"{fact.value} {fact.detail}".strip(), candidates=candidates).choice.strip()
        # Step 2 check: the answer must be one of the candidates, copied exactly.
        return choice if choice in candidates else None

    def forward(self, turns: list[Turn], language: str = "en-US", recorded_vitals: str = "none") -> dspy.Prediction:
        extracted = self.facts(turns=turns)
        facts: list[Fact] = extracted.facts
        matches = {f.id: m for f in facts if (m := self.match(f))}
        written = self.write(facts=facts, recorded_vitals=recorded_vitals, language=language)
        note = {s: getattr(written, s) for s in SOAP_SECTIONS}
        uncited, missing = check_note(note, facts)
        return dspy.Prediction(
            facts=facts,
            rejected_facts=extracted.rejected,
            catalog_matches=matches,
            note=note,
            uncited_sentences=uncited,
            facts_missing_from_note=missing,
        )
