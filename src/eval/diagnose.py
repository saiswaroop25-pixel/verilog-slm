"""Part 6: the offline diagnostic pass. Takes M0's probe-split generations
(from src.infer.generate) and produces the taxonomy table that both
(a) stands as a deliverable on its own, and (b) feeds
train/curriculum.py's compute_reweighting() for M1.

    python -m src.infer.generate --adapter artifacts/m0/final --split probe \\
        --n 5 --temperature 0.8 --top_p 0.95 --out artifacts/m0_probe_gens.jsonl
    python -m src.eval.diagnose \\
        --gens artifacts/m0_probe_gens.jsonl --out artifacts/m0_diagnostic.json

Note: the probe split comes from RTLCoder/MG-Verilog, which ship
instruction->code pairs with no testbench, so this pass uses
verify_static() (compile+lint+synthesize only) rather than the full
verify() -- there's no oracle to simulate against here. Functional
(simulate-stage) failure rates are measured separately, on the eval sets,
inside run_eval.py where real testbenches exist.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.eval.metrics import catch_all_share, construct_failure_rate_table, taxonomy_table
from src.infer.postprocess import finalize
from src.utils.io_utils import read_jsonl
from src.verify.taxonomy import CATCH_ALL_LABELS


def _verify_one(args: tuple) -> dict[str, Any]:
    from src.verify.harness import verify_static
    code, tier, constructs = args
    result = verify_static(code, top_module="top")
    return {
        "stage": result.stage, "ok": result.ok, "error_label": result.error_label,
        "tier": tier, "constructs": constructs,
    }


def run(gens_path: str, out_path: str) -> None:
    rows = list(read_jsonl(gens_path))

    verify_args = []
    for row in rows:
        tier = row["tags"]["tier"]
        constructs = row["tags"].get("constructs", [])
        for sample in row["samples"]:
            verify_args.append((finalize(sample), tier, constructs))

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_verify_one, verify_args))

    table = taxonomy_table(results)
    construct_rates = construct_failure_rate_table(results)
    report = {
        "n_problems": len(rows),
        "n_generations": len(results),
        "labels": table,
        "catch_all_share": catch_all_share(table, CATCH_ALL_LABELS),
        # Keyed by construct tag, not error label -- this is what
        # train/curriculum.py's compute_reweighting() actually consumes
        # for M1 (see construct_failure_rate_table's docstring for why
        # `labels` above can't be used for that directly).
        "construct_failure_rates": construct_rates,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[diagnose] {report['n_generations']} generations, "
          f"catch-all share (wrong_logic_other + *_other) = {report['catch_all_share']:.1%}")
    if report["catch_all_share"] > 0.6:
        print("[diagnose] WARNING: catch-all share exceeds 60% -- the taxonomy is not "
              "discriminating well; see Part 13's row on this before building the "
              "curriculum on top of it.")
    top_constructs = list(construct_rates.items())[:5]
    print("[diagnose] highest-failure-rate constructs: " +
          ", ".join(f"{tag}={v['fail_rate']:.0%} (n={v['count']})" for tag, v in top_constructs))
    print(f"[diagnose] wrote -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.gens, args.out)


if __name__ == "__main__":
    main()
