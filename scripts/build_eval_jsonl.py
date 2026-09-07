"""Adapter: turn a cloned VerilogEval or RTLLM checkout into this repo's
eval jsonl schema (Part 3's "eval only" sources).

Target schema, one row per problem:
    {
      "id": "...", "instruction": "...", "code": "<reference solution>",
      "testbench": "<the benchmark's own testbench>",
      "top_module": "...", "tags": {"tier": "T1".."T4"}
    }

VerilogEval and RTLLM both ship one directory per problem containing a
spec/prompt file, a reference solution, and a testbench, but the exact
filenames differ across release tags of each repo (both have reorganised
their layout at least once). Rather than hardcode a glob that silently
finds zero files against whatever tag you happen to `git clone`, this
script fails loudly and tells you what it found, so you fix the three
glob patterns below once against the tag you're actually using and move
on -- a wrong eval set silently scored as empty is a much worse failure
mode than an ImportError-style crash here.

    python -m scripts.build_eval_jsonl --src /tmp/verilogeval --out data/eval/verilogeval_v2.jsonl --benchmark verilogeval
    python -m scripts.build_eval_jsonl --src /tmp/rtllm --out data/eval/rtllm_v2.jsonl --benchmark rtllm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.tagger import tag_example
from src.utils.io_utils import write_jsonl

# Adjust these three per the exact repo tag you cloned -- see module
# docstring. Left as an explicit, greppable config block rather than
# buried string literals.
GLOB_PATTERNS = {
    "verilogeval": {
        "problem_dirs": "**/Prob*",
        "spec_file": "prompt.txt",
        "ref_file": "ref.sv",
        "tb_file": "testbench.sv",
    },
    "rtllm": {
        "problem_dirs": "*/",
        "spec_file": "design_description.txt",
        "ref_file": "*.v",
        "tb_file": "testbench.v",
    },
}


def build(src: str, out: str, benchmark: str) -> None:
    patterns = GLOB_PATTERNS[benchmark]
    src_path = Path(src)
    problem_dirs = sorted(src_path.glob(patterns["problem_dirs"]))

    if not problem_dirs:
        raise SystemExit(
            f"[build_eval_jsonl] found 0 problem directories under {src} matching "
            f"'{patterns['problem_dirs']}'. The upstream repo layout has likely "
            f"changed since GLOB_PATTERNS['{benchmark}'] was written -- inspect "
            f"`find {src} -maxdepth 3` and fix the patterns at the top of this file."
        )

    rows = []
    n_missing = 0
    for pdir in problem_dirs:
        if not pdir.is_dir():
            continue
        spec_matches = list(pdir.glob(patterns["spec_file"]))
        ref_matches = list(pdir.glob(patterns["ref_file"]))
        tb_matches = list(pdir.glob(patterns["tb_file"]))
        if not (spec_matches and ref_matches and tb_matches):
            n_missing += 1
            continue

        code = ref_matches[0].read_text(encoding="utf-8", errors="ignore")
        tags = tag_example(code)
        top_module_match = None
        import re
        m = re.search(r"\bmodule\s+(\w+)", code)
        top_module = m.group(1) if m else "top"

        rows.append({
            "id": f"{benchmark}_{pdir.name}",
            "instruction": spec_matches[0].read_text(encoding="utf-8", errors="ignore").strip(),
            "code": code,
            "testbench": tb_matches[0].read_text(encoding="utf-8", errors="ignore"),
            "top_module": top_module,
            "tags": {"tier": tags.tier},
        })

    if n_missing:
        print(f"[build_eval_jsonl] WARNING: {n_missing}/{len(problem_dirs)} problem dirs "
              f"were missing one of spec/ref/testbench and were skipped.")

    write_jsonl(out, rows)
    print(f"[build_eval_jsonl] wrote {len(rows)} problems -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", required=True, choices=list(GLOB_PATTERNS))
    args = ap.parse_args()
    build(args.src, args.out, args.benchmark)


if __name__ == "__main__":
    main()
