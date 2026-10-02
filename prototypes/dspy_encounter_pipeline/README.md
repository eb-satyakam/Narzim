# DSPy encounter pipeline — prototype

A small working sketch of the encounter AI as a **pipeline of steps** instead of
one big prompt, with **DSPy** used to tune each step's prompt against an answer
key. Standalone: it does not touch the Frappe app, and DSPy never runs in
production.

## The idea in one picture

```
                         ┌──────────── development (DSPy) ────────────┐
 answer-key cases ──► optimize.py ──► score ─► tune ─► score ─► export │
                         └────────────────────────────────────┬───────┘
                                                              ▼
                                   prompts/fact-extract-dspy-v1.txt  (+ .messages.json, .meta.json)
                                                              │  reviewed in a PR like any prompt
 ┌──────────────────────── production (no DSPy) ──────────────▼───────────────────┐
 │ transcript ─► 1 Extract facts ─► check ─► 2 Catalog match ─► check ─►          │
 │ (turn ids)    reads transcript    every     code finds top 5,   answer must     │
 │               once                fact      small model picks   be a candidate │
 │                                   cites                                         │
 │                                   real                                          │
 │                                   words                                         │
 │            ─► 3 Write SOAP from facts only ─► check: every sentence cites a     │
 │                                               fact; unused facts = gaps         │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

**DSPy writes the prompt; your app runs it.** After optimizing, `export.py`
writes the prompt as plain files. The Frappe adapter loads them like every
other prompt in `prompts/`, so prompts stay files, stay versioned, get hashed
into `AI Prompt Log`, and get reviewed in a PR. `runtime.py` shows the
production side: fill the placeholders, call OpenAI or Anthropic as today,
validate the reply with Pydantic. It does not import DSPy.

## Files

| File | What it is |
|---|---|
| `encounter_pipeline/models.py` | The typed shapes passed between steps (`Fact`, `NoteSentence`). Pydantic only |
| `encounter_pipeline/signatures.py` | Each AI step's contract. The docstring is the instruction DSPy tunes |
| `encounter_pipeline/pipeline.py` | The steps, and the code checks between them |
| `encounter_pipeline/metrics.py` | How a fact extraction is scored. A flipped negation scores the case 0 |
| `encounter_pipeline/export.py` | Optimized step → `.txt` / `.messages.json` / `.meta.json` |
| `encounter_pipeline/runtime.py` | Running an exported step with no DSPy installed |
| `data/fact_cases.jsonl` | 8 made-up consultations with answer keys (5 train, 3 dev) |
| `optimize.py` | Score → tune → score → export |
| `tests/` | Runs with DSPy's fake model, no API key needed |

## Run it

```bash
cd prototypes/dspy_encounter_pipeline
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

pytest                                   # no key needed

export OPENAI_API_KEY=...
python optimize.py                       # adds worked examples, keeps the instruction
python optimize.py --optimizer mipro     # also rewrites the instruction (needs more cases)
```

`optimize.py` prints the dev score before and after, and refuses to export a
version that scores worse than the one it started from.

## What the code checks catch, whatever the prompt says

| Check | Catches |
|---|---|
| A fact's turns must exist and contain its quote | Invented findings |
| A catalog pick must be one of the 5 candidates code found | Made-up drug or diagnosis names |
| Every note sentence must cite a fact | Statements nothing in the visit supports |
| Facts no sentence cites | Gaps: said in the room, missing from the note |

## Before this goes near real use

- **Cases.** 8 is enough to see it work, not to trust it. MIPRO wants roughly
  50+ per step. Write them with a clinician, and keep them made up: real
  transcripts are patient data and do not belong in a repo or on a laptop.
- **Metric.** Matching is by substring ("throat" matches "red throat"). Fine
  for a sketch; a real one needs synonyms or a second model as judge.
- **Fact storage.** In the app the fact list is a new copy of patient data:
  it needs an erasure and retention path (PHI-4), a migration patch (BE-9),
  and must stay out of logs (BE-10).
- **Dependency.** DSPy is a development tool here only. Using it inside the
  Frappe app would be a dependency decision for an ADR.
- **Incremental facts during recording** (only new turns, merged into the
  running list) are not in this sketch.
