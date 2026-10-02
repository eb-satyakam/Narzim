"""Score the fact-extraction step, let DSPy improve it, score again, export.

    export OPENAI_API_KEY=...
    python optimize.py                                  # quick: worked examples only
    python optimize.py --optimizer mipro                # also rewrites the instruction
    python optimize.py --model anthropic/claude-sonnet-5-5

Run it on made-up cases only (data/fact_cases.jsonl). Real transcripts are
patient data and do not belong on a laptop or in a repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dspy

from encounter_pipeline.data import load_cases
from encounter_pipeline.export import export_step
from encounter_pipeline.metrics import fact_metric
from encounter_pipeline.pipeline import FactExtractor


def evaluate(program: dspy.Module, devset: list[dspy.Example]) -> float:
    result = dspy.Evaluate(devset=devset, metric=fact_metric, num_threads=4, display_table=True)(program)
    return float(result.score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--optimizer", choices=("bootstrap", "mipro"), default="bootstrap")
    parser.add_argument("--version", default="fact-extract-dspy-v1", help="Name of the exported prompt files.")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM(args.model, temperature=0.0))
    train, dev = load_cases()

    baseline = FactExtractor()
    before = evaluate(baseline, dev)

    if args.optimizer == "mipro":
        optimizer = dspy.MIPROv2(metric=fact_metric, auto="light", num_threads=4)
        # minibatch off: the starter set is far smaller than MIPRO's default batch of 35.
        tuned = optimizer.compile(
            baseline.deepcopy(), trainset=train, minibatch=False, requires_permission_to_run=False
        )
    else:
        optimizer = dspy.BootstrapFewShot(metric=fact_metric, max_bootstrapped_demos=3, max_labeled_demos=0)
        tuned = optimizer.compile(baseline.deepcopy(), trainset=train)
    after = evaluate(tuned, dev)

    print(f"\ndev score: {before:.1f} -> {after:.1f}")
    if after < before:
        print("Tuned version is worse on dev; not exporting.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    tuned.save(args.out / f"{args.version}.dspy.json")
    paths = export_step(
        tuned.extract, args.version, args.out, score=after, model=args.model, optimizer=args.optimizer
    )
    for kind, path in paths.items():
        print(f"{kind:9} {path}")


if __name__ == "__main__":
    main()
