"""Quick pass: generate + verify against the eval sets (real testbenches),
producing the verify-results jsonl that scripts/validate_classifier.py
samples from for the Part 4 hand-label check.

Deliberately separate from src/eval/run_eval.py's full 2x2 driver -- this
is a one-model, one-seed, small-n pass whose only job is to produce
enough failures with real ground truth to hand-label 100 of them.

    python -m scripts.build_eval_verify_results \\
        --adapter artifacts/m0/final --eval-sets data/eval/verilogeval_v2.jsonl data/eval/rtllm_v2.jsonl \\
        --n 3 --out artifacts/eval_verify_results.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from src.infer.generate import generate_batch, load_model_for_inference
from src.infer.postprocess import finalize
from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.prompts import build_prompt


def _verify_one(args: tuple) -> dict:
    from src.verify.harness import verify
    code, testbench, top_module, tier, problem_id = args
    result = verify(code, testbench, top_module=top_module)
    return {
        "problem_id": problem_id, "stage": result.stage, "ok": result.ok,
        "error_label": result.error_label, "tier": tier, "error_text": result.error_text,
    }


def run(adapter: str, eval_paths: list[str], n: int, out_path: str) -> None:
    eval_rows = []
    for p in eval_paths:
        eval_rows.extend(read_jsonl(p))

    model, tokenizer = load_model_for_inference(adapter)
    prompts = [build_prompt(r["instruction"]) for r in eval_rows]
    completions = generate_batch(model, tokenizer, prompts, n=n, temperature=0.8, top_p=0.95)

    verify_args = []
    for row, comps in zip(eval_rows, completions):
        for code in comps:
            verify_args.append((finalize(code), row["testbench"], row.get("top_module", "top"),
                                 row["tags"]["tier"], row["id"]))

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_verify_one, verify_args))

    write_jsonl(out_path, results)
    n_fail = sum(1 for r in results if not r["ok"])
    print(f"[build_eval_verify_results] {len(results)} generations, {n_fail} failures -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval-sets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.adapter, args.eval_sets, args.n, args.out)


if __name__ == "__main__":
    main()
