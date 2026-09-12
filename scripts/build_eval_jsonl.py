"""Adapter: turn a cloned VerilogEval or RTLLM checkout into this repo's
eval jsonl schema (Part 3's "eval only" sources).

Target schema, one row per problem:
    {
      "id": "...", "instruction": "...", "code": "<reference solution>",
      "testbench": "<what gets compiled alongside the model's generated
                     code as testbench.v -- see per-benchmark notes below>",
      "top_module": "...", "tags": {"tier": "T1".."T4"}
    }

Both benchmarks' actual layout and testbench conventions were inspected
directly against a live clone before writing this (not guessed from
memory) -- an earlier version of this script guessed wrong on almost
every count, so the notes below are load-bearing, not decorative.

    python -m scripts.build_eval_jsonl --src /tmp/verilogeval --out data/eval/verilogeval_v2.jsonl --benchmark verilogeval
    python -m scripts.build_eval_jsonl --src /tmp/rtllm --out data/eval/rtllm_v2.jsonl --benchmark rtllm
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

from src.data.tagger import tag_example
from src.utils.io_utils import write_jsonl

_RTLLM_MODULE_NAME_RE = re.compile(r"[Mm]odule\s+[Nn]ame\s*:\s*\n?\s*(\w+)")


def build_verilogeval(src: str, out: str) -> None:
    """VerilogEval v2's `dataset_spec-to-rtl/` directory is FLAT -- there
    are no per-problem subdirectories. Each problem is three files sharing
    a prefix: `<prefix>_prompt.txt`, `<prefix>_ref.sv`, `<prefix>_test.sv`.

    Every prompt asks the model to implement a module literally named
    `TopModule` (fixed across the whole benchmark). The testbench does
    DIFFERENTIAL testing: it instantiates the reference solution under a
    *different* name, `RefModule` (from `_ref.sv`), side by side with the
    model's `TopModule`, and compares their outputs -- so `_ref.sv` must be
    compiled together with the testbench, not treated as a separate
    "reference only" artifact. Concatenating `_ref.sv` + `_test.sv` into
    this row's "testbench" field reproduces that with zero changes needed
    to the harness's plain `iverilog design.v testbench.v` compile step.

    Pass/fail: `$display("Mismatches: %1d in %1d samples", ...)` -- already
    matched by the harness's existing "Mismatches: N" parsing.
    """
    data_dir = Path(src) / "dataset_spec-to-rtl"
    prompt_files = sorted(data_dir.glob("*_prompt.txt"))
    if not prompt_files:
        raise SystemExit(
            f"[build_eval_jsonl] found 0 '*_prompt.txt' files under {data_dir}. "
            f"VerilogEval has reorganised this layout before -- inspect "
            f"`find {src} -maxdepth 2` and update build_verilogeval() to match."
        )

    rows = []
    n_missing = 0
    for prompt_path in prompt_files:
        prefix = prompt_path.name[: -len("_prompt.txt")]
        ref_path = data_dir / f"{prefix}_ref.sv"
        test_path = data_dir / f"{prefix}_test.sv"
        if not (ref_path.exists() and test_path.exists()):
            n_missing += 1
            continue

        ref_code = ref_path.read_text(encoding="utf-8", errors="ignore")
        test_code = test_path.read_text(encoding="utf-8", errors="ignore")
        instruction = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
        tier = tag_example(ref_code).tier

        rows.append({
            "id": f"verilogeval_{prefix}",
            "instruction": instruction,
            "code": ref_code,
            "testbench": ref_code + "\n" + test_code,
            "top_module": "TopModule",
            "tags": {"tier": tier},
        })

    if n_missing:
        print(f"[build_eval_jsonl] WARNING: {n_missing} prompt files were missing "
              f"a matching _ref.sv/_test.sv and were skipped.")

    write_jsonl(out, rows)
    print(f"[build_eval_jsonl] wrote {len(rows)} problems -> {out}")


def build_rtllm(src: str, out: str) -> None:
    """RTLLM v2 is organised as Category/Subcategory/<problem>/, nested to
    varying depth -- discover problems by the presence of
    `design_description.txt` rather than assuming a fixed depth.

    Each problem dir has: `design_description.txt` (spec, which states the
    required module name as "Module name:\\n    <name>"), `testbench.v`
    (a SELF-CHECKING testbench -- it computes the expected result itself
    and instantiates the model's module directly under the name from
    design_description.txt; no separate reference file is needed at
    compile time), and `verified_*.v` (the reference solution, used here
    only for tagging/dedup -- its internal module name is a differently-
    prefixed name than what the model must produce, e.g. `adder_16bit`
    (required) vs. `verified_adder_16bit` (reference file's own name), so
    it is NOT concatenated into the testbench the way VerilogEval's is).

    Pass/fail sentinel: "===========Your Design Passed==========="
    (dash count and spacing vary per problem) -- confirmed present in all
    50/50 of this benchmark's testbenches; matched by the harness with
    flexible whitespace, not a literal string compare. Failure messages
    are NOT standardized across problems, so absence of the pass sentinel
    is what the harness treats as failure.
    """
    desc_paths = sorted(glob.glob(os.path.join(src, "**", "design_description.txt"), recursive=True))
    if not desc_paths:
        raise SystemExit(
            f"[build_eval_jsonl] found 0 'design_description.txt' files under {src}. "
            f"RTLLM has reorganised this layout before -- inspect "
            f"`find {src} -name design_description.txt` and update build_rtllm() to match."
        )

    rows = []
    n_missing = 0
    for desc_path_str in desc_paths:
        desc_path = Path(desc_path_str)
        problem_dir = desc_path.parent
        tb_path = problem_dir / "testbench.v"
        ref_matches = list(problem_dir.glob("verified_*.v"))
        if not (tb_path.exists() and ref_matches):
            n_missing += 1
            continue

        instruction = desc_path.read_text(encoding="utf-8", errors="ignore").strip()
        name_match = _RTLLM_MODULE_NAME_RE.search(instruction)
        if not name_match:
            n_missing += 1
            continue
        top_module = name_match.group(1)

        ref_code = ref_matches[0].read_text(encoding="utf-8", errors="ignore")
        testbench = tb_path.read_text(encoding="utf-8", errors="ignore")
        tier = tag_example(ref_code).tier

        rows.append({
            "id": f"rtllm_{problem_dir.name}",
            "instruction": instruction,
            "code": ref_code,
            "testbench": testbench,
            "top_module": top_module,
            "tags": {"tier": tier},
        })

    if n_missing:
        print(f"[build_eval_jsonl] WARNING: {n_missing}/{len(desc_paths)} problem dirs "
              f"were missing testbench.v/verified_*.v or a parseable module name, "
              f"and were skipped.")

    write_jsonl(out, rows)
    print(f"[build_eval_jsonl] wrote {len(rows)} problems -> {out}")


BUILDERS = {"verilogeval": build_verilogeval, "rtllm": build_rtllm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", required=True, choices=list(BUILDERS))
    args = ap.parse_args()
    BUILDERS[args.benchmark](args.src, args.out)


if __name__ == "__main__":
    main()
