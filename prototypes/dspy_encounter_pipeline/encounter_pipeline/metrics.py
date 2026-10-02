"""How a fact extraction is scored. The optimizer climbs this number.

The metric is the real clinical rule set, so it is written to be strict where
the clinic would be: a flipped negation scores the whole case zero, because
"fever" and "no fever" in a note are not a small difference.
"""

from __future__ import annotations

import dspy

from .pipeline import _norm


def _same_finding(gold: dict, fact) -> bool:
    g, p = _norm(gold["value"]), _norm(fact.value)
    return bool(g and p) and (g in p or p in g)


def fact_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float | bool:
    """Recall of the answer key, minus a penalty for invented facts.

    - Any gold fact found with its negation flipped -> 0.0 for the case.
    - Score = share of gold facts found, minus 0.5 x share of facts the
      grounding check rejected (the model cited words nobody said).
    - While bootstrapping worked examples (trace is set) only a near-perfect
      run counts, so the examples DSPy keeps are ones worth copying.
    """
    gold = example.gold_facts
    facts = list(pred.facts)
    found = 0
    for g in gold:
        hits = [f for f in facts if _same_finding(g, f)]
        if any(f.negated != g.get("negated", False) for f in hits):
            return False if trace is not None else 0.0
        if hits:
            found += 1
    recall = found / len(gold) if gold else 1.0
    produced = len(facts) + len(pred.rejected)
    invented = len(pred.rejected) / produced if produced else 0.0
    score = max(0.0, recall - 0.5 * invented)
    return score >= 0.9 if trace is not None else score
