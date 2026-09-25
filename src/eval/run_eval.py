"""The 2x2 driver (Part 10).

              | Repair off | Repair on |
    M0        |     A      |     B     |
    M1        |     C      |     D     |

    python -m src.eval.run_eval \\
        --m0-adapter artifacts/m0/final --m1-adapter artifacts/m1/final \\
        --eval-sets data/eval/verilogeval_v2.jsonl data/eval/rtllm_v2.jsonl \\
        --n 20 --k-repair 3 --seeds 1337 2025 7 \\
        --out artifacts/eval_report.json

Each eval-set row is expected to carry: id, instruction, testbench,
top_module, tier (structural tier, for the taxonomy table), and code
(reference solution, used only for corpus-contamination checks upstream --
never used for scoring here; scoring is verification-gated).

Runs simulation in a process pool (Part 4: simulation is CPU-bound and
embarrassingly parallel -- run this cell on a CPU-only Colab runtime to
avoid burning GPU units on iverilog/vvp).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.eval.metrics import (
    accuracy_per_gpu_hour,
    bootstrap_ci,
    catch_all_share,
    compute_pass_at_k_per_problem,
    repair_diagnostics,
    syntax_error_rate,
    taxonomy_table,
)
from src.infer.generate import generate_batch, load_model_for_inference
from src.infer.repair import generate_with_repair, make_hf_generate_fn
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.seeding import set_seed
from src.verify.taxonomy import CATCH_ALL_LABELS

TARGETED_LABELS = {"incomplete_sensitivity", "missing_default_case"}  # 3x weight in curriculum


def _verify_one(args: tuple) -> dict[str, Any]:
    from src.verify.harness import verify
    code, testbench, top_module, tier = args
    result = verify(code, testbench, top_module=top_module)
    return {
        "stage": result.stage, "ok": result.ok, "error_label": result.error_label,
        "tier": tier, "wall_ms": result.wall_ms,
    }


def run_no_repair_cell(model, tokenizer, eval_rows: list[dict[str, Any]], n: int,
                        max_new_tokens: int = 768) -> list[dict[str, Any]]:
    from src.utils.prompts import build_prompt

    prompts = [build_prompt(r["instruction"]) for r in eval_rows]
    completions = generate_batch(model, tokenizer, prompts, n=n, temperature=0.8, top_p=0.95,
                                  max_new_tokens=max_new_tokens)

    verify_args = []
    problem_index = []
    for row, comps in zip(eval_rows, completions):
        for code in comps:
            from src.infer.postprocess import finalize
            verify_args.append((finalize(code), row["testbench"], row.get("top_module", "top"), row["tier"]))
            problem_index.append(row["id"])

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_verify_one, verify_args))

    for r, pid in zip(results, problem_index):
        r["problem_id"] = pid
    return results


def run_repair_cell(model, tokenizer, eval_rows: list[dict[str, Any]], k_repair: int,
                     max_new_tokens: int = 768) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import time

    # Unlike run_no_repair_cell, this isn't batched -- each repair attempt
    # depends on the previous one's error, so problems are processed one
    # at a time, each up to K+1 sequential generate() calls. No progress
    # output otherwise made this indistinguishable from a hang for
    # anything but a handful of problems.
    gen_fn = make_hf_generate_fn(model, tokenizer, max_new_tokens=max_new_tokens)
    verify_results, traces = [], []
    start_time = time.monotonic()
    for i, row in enumerate(eval_rows):
        code, ok, trace = generate_with_repair(
            gen_fn, row["instruction"], row["testbench"],
            top_module=row.get("top_module", "top"), K=k_repair,
        )
        last = trace.attempts[-1]
        verify_results.append({
            "stage": last["stage"], "ok": ok, "error_label": last["label"],
            "tier": row["tier"], "problem_id": row["id"],
        })
        traces.append({"attempts": trace.attempts, "regressions": trace.regressions, "problem_id": row["id"]})

        elapsed = time.monotonic() - start_time
        done = i + 1
        eta_min = (elapsed / done) * (len(eval_rows) - done) / 60
        print(f"[run_eval] repair problem {done}/{len(eval_rows)} "
              f"({len(trace.attempts)} attempts, {elapsed/60:.1f}m elapsed, ~{eta_min:.1f}m remaining)")
    return verify_results, traces


def summarise_cell(verify_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_problem: dict[str, list[bool]] = {}
    for r in verify_results:
        by_problem.setdefault(r["problem_id"], []).append(r["ok"])

    pass_lists = compute_pass_at_k_per_problem(list(by_problem.values()), ks=[1, 5])
    summary = {"n_generations": len(verify_results), "n_problems": len(by_problem)}
    for k, vals in pass_lists.items():
        mean, lo, hi = bootstrap_ci(vals)
        summary[f"pass@{k}"] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}
    summary["syntax_error_rate"] = syntax_error_rate(verify_results)
    table = taxonomy_table(verify_results)
    summary["taxonomy_table"] = table
    summary["catch_all_share"] = catch_all_share(table, CATCH_ALL_LABELS)
    return summary


def run_2x2(m0_adapter: str, m1_adapter: str, eval_rows: list[dict[str, Any]], n: int, k_repair: int,
            max_new_tokens: int = 768) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for model_key, adapter in [("M0", m0_adapter), ("M1", m1_adapter)]:
        print(f"[run_eval] loading {model_key} ({adapter})...")
        model, tokenizer = load_model_for_inference(adapter)

        print(f"[run_eval] {model_key}: repair-off cell ({len(eval_rows)} problems x {n} samples)...")
        no_repair = run_no_repair_cell(model, tokenizer, eval_rows, n, max_new_tokens=max_new_tokens)
        cells[f"{model_key}_repair_off"] = summarise_cell(no_repair)

        print(f"[run_eval] {model_key}: repair-on cell ({len(eval_rows)} problems, up to {k_repair} repairs each)...")
        with_repair, traces = run_repair_cell(model, tokenizer, eval_rows, k_repair, max_new_tokens=max_new_tokens)
        cell = summarise_cell(with_repair)
        cell["repair_diagnostics"] = repair_diagnostics(traces)
        cell["budget"] = f"1 sample, up to {k_repair} repairs"
        cells[f"{model_key}_repair_on"] = cell

        del model
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    a = cells["M0_repair_off"]["pass@1"]["mean"]
    b = cells["M0_repair_on"]["pass@1"]["mean"]
    c = cells["M1_repair_off"]["pass@1"]["mean"]
    d = cells["M1_repair_on"]["pass@1"]["mean"]
    effects = {
        "curriculum_effect_C_minus_A": c - a,
        "repair_effect_B_minus_A": b - a,
        "interaction_D_minus_C_minus_B_minus_A": (d - c) - (b - a),
    }

    return {"cells": cells, "effects": effects}


def run_multi_seed(m0_adapter: str, m1_adapter: str, eval_paths: list[str], n: int,
                    k_repair: int, seeds: list[int], out_path: str,
                    m0_gpu_hours: float | None = None, m1_gpu_hours: float | None = None,
                    limit: int | None = None, max_new_tokens: int = 768) -> None:
    eval_rows: list[dict[str, Any]] = []
    for p in eval_paths:
        eval_rows.extend(read_jsonl(p))

    total_available = len(eval_rows)
    if limit is not None and limit < total_available:
        # A fixed sampling seed (not one of the eval `seeds`) so the same
        # subset of problems is used across every seed and every cell --
        # only the generation randomness should vary per seed, not which
        # problems get evaluated.
        import random
        eval_rows = random.Random(1337).sample(eval_rows, limit)
        print(f"[run_eval] evaluating a random subsample of {limit} problems (out of {total_available})")

    per_seed_results = []
    for seed in seeds:
        set_seed(seed)
        print(f"[run_eval] seed={seed}")
        per_seed_results.append(run_2x2(m0_adapter, m1_adapter, eval_rows, n, k_repair, max_new_tokens=max_new_tokens))

    report = {
        "seeds": seeds, "n_eval_problems": len(eval_rows),
        "per_seed": per_seed_results,
    }

    if m0_gpu_hours and m1_gpu_hours:
        a1 = per_seed_results[0]["cells"]["M0_repair_off"]["pass@1"]["mean"]
        c1 = per_seed_results[0]["cells"]["M1_repair_off"]["pass@1"]["mean"]
        report["efficiency"] = {
            "M0_pass1_per_gpu_hour": accuracy_per_gpu_hour(a1, m0_gpu_hours),
            "M1_pass1_per_gpu_hour": accuracy_per_gpu_hour(c1, m1_gpu_hours),
            "note": "GPU-hours across different GPU models are not directly "
                    "comparable; report GPU model alongside. Published "
                    "baselines (e.g. RTLCoder) don't report training cost, "
                    "so this comparison is one-sided by construction.",
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[run_eval] wrote report -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m0-adapter", required=True)
    ap.add_argument("--m1-adapter", required=True)
    ap.add_argument("--eval-sets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k-repair", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2025, 7])
    ap.add_argument("--out", required=True)
    ap.add_argument("--m0-gpu-hours", type=float, default=None)
    ap.add_argument("--m1-gpu-hours", type=float, default=None)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of eval problems (random subsample, same fixed seed "
             "across --seeds so every seed evaluates the same problems) -- the full "
             "combined eval sets can be a few hundred problems, and each one costs "
             "n generations (repair-off) plus up to k_repair+1 more (repair-on), "
             "per model, per seed",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="passed through to both the repair-off (batched) and repair-on "
             "(one-at-a-time) generation paths",
    )
    args = ap.parse_args()

    run_multi_seed(
        args.m0_adapter, args.m1_adapter, args.eval_sets, args.n, args.k_repair,
        args.seeds, args.out, args.m0_gpu_hours, args.m1_gpu_hours,
        args.limit, args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
