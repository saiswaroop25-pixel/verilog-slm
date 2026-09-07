"""Validate the error classifier against hand labels before trusting it
(Part 4: "Hand-label 100 random failures. Compute classifier agreement.
If it's below ~85%, fix the rules before you build a curriculum on top of
it."). This number belongs in the report.

Workflow:
  1. Run generations + verify(), collect failing records.
  2. `python -m scripts.validate_classifier sample --results artifacts/m0_probe_verify.jsonl \\
        --n 100 --out artifacts/hand_label_sample.jsonl`
     -> writes 100 random failures with an empty "human_label" field.
  3. Open the file, fill in "human_label" for each row (use the exact
     ErrorLabel values from src/verify/taxonomy.py).
  4. `python -m scripts.validate_classifier score --labeled artifacts/hand_label_sample.jsonl`
     -> prints agreement rate and a confusion breakdown.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from src.utils.io_utils import read_jsonl, write_jsonl


def sample(results_path: str, n: int, out_path: str, seed: int = 1337) -> None:
    rows = [r for r in read_jsonl(results_path) if not r.get("ok", r.get("stage") != "pass")]
    if len(rows) < n:
        print(f"[validate_classifier] WARNING: only {len(rows)} failures available, requested {n}")
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(n, len(rows)))
    for row in chosen:
        row["human_label"] = ""
    write_jsonl(out_path, chosen)
    print(f"[validate_classifier] wrote {len(chosen)} rows -> {out_path}. "
          f"Fill in 'human_label' for each, then run the 'score' subcommand.")


def score(labeled_path: str) -> None:
    rows = list(read_jsonl(labeled_path))
    unlabeled = [r for r in rows if not r.get("human_label")]
    if unlabeled:
        print(f"[validate_classifier] {len(unlabeled)} rows still have an empty human_label -- "
              f"fill those in before scoring.")
        return

    agree = sum(1 for r in rows if r["human_label"] == r["error_label"])
    rate = agree / len(rows) if rows else 0.0

    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        confusion[r["human_label"]][r["error_label"]] += 1

    print(f"[validate_classifier] agreement: {agree}/{len(rows)} = {rate:.1%}")
    if rate < 0.85:
        print("[validate_classifier] WARNING: below the 85% target -- fix classify.py rules "
              "before building the curriculum reweighting on top of this taxonomy (Part 4).")

    print("\nconfusion (human_label -> {classifier_label: count}):")
    for human_label, counts in sorted(confusion.items()):
        print(f"  {human_label}: {dict(counts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--results", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=1337)

    c = sub.add_parser("score")
    c.add_argument("--labeled", required=True)

    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.results, args.n, args.out, args.seed)
    elif args.cmd == "score":
        score(args.labeled)


if __name__ == "__main__":
    main()
