"""Runs without an API key: DSPy's DummyLM plays the model."""

from __future__ import annotations

import json

import dspy
import pytest
from dspy.utils import DummyLM

from encounter_pipeline.data import load_cases
from encounter_pipeline.export import export_step
from encounter_pipeline.metrics import fact_metric
from encounter_pipeline.models import Fact, NoteSentence
from encounter_pipeline.pipeline import (
    EncounterPipeline,
    FactExtractor,
    Turn,
    catalog_candidates,
    check_note,
    ground_facts,
)
from encounter_pipeline.runtime import parse_facts, render

TURNS = [
    Turn("t01", "doctor", "Any fever?"),
    Turn("t02", "patient", "No fever, but a cough for three days."),
    Turn("t03", "doctor", "Start amoxicillin 500 mg three times a day."),
]


def fact(value, quote, turns, **kw) -> dict:
    base = {"type": "symptom", "value": value, "speaker": "patient", "certainty": "stated", "turns": turns, "quote": quote}
    return {**base, **kw}


GOOD = [
    fact("fever", "No fever", ["t02"], negated=True),
    fact("cough", "a cough for three days", ["t02"], detail="three days"),
    fact("amoxicillin 500 mg", "amoxicillin 500 mg", ["t03"], type="medication", speaker="doctor"),
]
INVENTED = fact("chest pain", "chest pain", ["t02"])  # nobody said it
BAD_TURN = fact("cough", "cough", ["t99"])  # cites a turn that does not exist


@pytest.fixture(autouse=True)
def _reset_lm():
    yield
    dspy.configure(lm=None)


# ── Code checks ──────────────────────────────────────────────────────────────


def test_grounding_rejects_invented_facts_and_missing_turns():
    facts = [Fact(**f) for f in [*GOOD, INVENTED, BAD_TURN]]
    kept, rejected = ground_facts(facts, TURNS)
    assert [f.value for f in kept] == ["fever", "cough", "amoxicillin 500 mg"]
    assert [f.id for f in kept] == ["f1", "f2", "f3"]
    assert [f.value for f in rejected] == ["chest pain", "cough"]


def test_catalog_candidates_are_capped_and_relevant():
    catalog = [f"Drug {i}" for i in range(200)] + ["Amoxicillin", "Amoxicillin + Clavulanate", "Ampicillin"]
    found = catalog_candidates("amoxicillin 500 mg", catalog)
    assert len(found) <= 5
    assert "Amoxicillin" in found
    assert not any(c.startswith("Drug ") for c in found)


def test_check_note_finds_uncited_sentences_and_gaps():
    facts = [Fact(**f).model_copy(update={"id": f"f{i}"}) for i, f in enumerate(GOOD, 1)]
    note = {
        "subjective": [NoteSentence(text="Cough for three days, no fever.", fact_ids=["f1", "f2"])],
        "objective": [NoteSentence(text="Lungs clear.", fact_ids=[])],
        "assessment": [],
        "plan": [NoteSentence(text="Ibuprofen.", fact_ids=["f9"])],
    }
    uncited, missing = check_note(note, facts)
    assert uncited == ["objective: Lungs clear.", "plan: Ibuprofen."]
    assert missing == ["f3"]


# ── Metric ───────────────────────────────────────────────────────────────────


def _example(gold):
    return dspy.Example(turns=TURNS, gold_facts=gold).with_inputs("turns")


def _pred(facts, rejected=()):
    return dspy.Prediction(facts=[Fact(**f) for f in facts], rejected=[Fact(**f) for f in rejected])


def test_metric_flipped_negation_scores_zero():
    gold = [{"type": "symptom", "value": "fever", "negated": True}, {"type": "symptom", "value": "cough"}]
    flipped = [fact("fever", "fever", ["t01"], negated=False), GOOD[1]]
    assert fact_metric(_example(gold), _pred(flipped)) == 0.0


def test_metric_penalises_invented_facts():
    gold = [{"type": "symptom", "value": "fever", "negated": True}, {"type": "symptom", "value": "cough"}]
    assert fact_metric(_example(gold), _pred(GOOD)) == 1.0
    assert fact_metric(_example(gold), _pred(GOOD, rejected=[INVENTED])) == pytest.approx(1 - 0.5 * 0.25)


# ── Whole pipeline, model faked ──────────────────────────────────────────────


def test_pipeline_end_to_end_with_fake_model():
    replies = [
        {"facts": json.dumps([*GOOD, INVENTED])},  # step 1: one invented fact
        {"choice": "Amoxicillin"},  # step 2: catalog pick for f3
        {  # step 3: note, one sentence cites nothing
            "subjective": json.dumps([{"text": "Cough for three days. No fever.", "fact_ids": ["f1", "f2"]}]),
            "objective": json.dumps([{"text": "Chest is clear.", "fact_ids": []}]),
            "assessment": json.dumps([]),
            "plan": json.dumps([{"text": "Amoxicillin 500 mg three times a day.", "fact_ids": ["f3"]}]),
        },
    ]
    dspy.configure(lm=DummyLM(replies))
    pipeline = EncounterPipeline({"medications": ["Amoxicillin", "Ampicillin"], "diagnoses": []})
    out = pipeline(turns=TURNS)

    assert [f.id for f in out.facts] == ["f1", "f2", "f3"]
    assert [f.value for f in out.rejected_facts] == ["chest pain"]
    assert out.catalog_matches == {"f3": "Amoxicillin"}
    assert out.uncited_sentences == ["objective: Chest is clear."]
    assert out.facts_missing_from_note == []


def test_catalog_pick_outside_candidates_is_dropped():
    dspy.configure(lm=DummyLM([{"choice": "Amoxil Forte 875"}]))
    pipeline = EncounterPipeline({"medications": ["Amoxicillin"]})
    med = Fact(**GOOD[2]).model_copy(update={"id": "f3"})
    assert pipeline.match(med) is None


# ── Export, and running the export without DSPy ──────────────────────────────


def test_export_then_render_and_parse(tmp_path):
    dspy.configure(lm=DummyLM([{"facts": json.dumps(GOOD)}]))
    extractor = FactExtractor()
    extractor.extract.demos = [
        dspy.Example(transcript="t01 patient: No cough.", facts=[Fact(**fact("cough", "No cough", ["t01"], negated=True))])
    ]
    paths = export_step(extractor.extract, "fact-extract-dspy-v1", tmp_path, score=87.5, model="test", optimizer="bootstrap")

    assert "Keep ruled-out findings as negated facts" in paths["prompt"].read_text()
    meta = json.loads(paths["meta"].read_text())
    assert meta["worked_examples"] == 1 and meta["dev_score"] == 87.5 and len(meta["sha256"]) == 64

    messages = render(paths["messages"], transcript="t01 patient: {curly} braces survive")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "{curly} braces survive" in messages[-1]["content"]
    with pytest.raises(ValueError, match="transcript"):
        render(paths["messages"])

    reply = f"[[ ## facts ## ]]\n{json.dumps(GOOD)}\n\n[[ ## completed ## ]]"
    assert [f.value for f in parse_facts(reply)] == ["fever", "cough", "amoxicillin 500 mg"]


def test_starter_cases_load():
    train, dev = load_cases()
    assert len(train) == 5 and len(dev) == 3
    assert all(isinstance(t, Turn) for t in train[0].turns)
