"""Build the training corpus (Part 3 of the guide).

    python -m src.data.build_corpus \
        --sources rtlcoder=data/raw/rtlcoder.jsonl mg-verilog=data/raw/mg_verilog.jsonl \
        --eval-sets data/eval/verilogeval_v2.jsonl data/eval/rtllm_v2.jsonl \
        --out artifacts/corpus.jsonl

Steps, in order:
  1. normalise every source into one schema
  2. MinHash-LSH near-dup removal within the training pool (Jaccard 0.85)
  3. MinHash-LSH contamination check against the eval sets -- drop any
     training example that collides with an eval problem. This step is
     not optional: VerilogEval is derived from HDLBits and HDLBits-style
     problems are common in GitHub-scraped corpora like RTLCoder/MG-Verilog.
  4. tag every example (structural tier + construct tags)
  5. stratified train/probe split

Every drop is logged and counted -- "N near-dups removed, M eval-contaminated
examples removed" is itself a reportable finding per the guide.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from src.data.splits import assign_splits
from src.data.tagger import tag_example
from src.utils.io_utils import read_jsonl, write_jsonl

_WHITESPACE_RE = re.compile(r"\s+")
_COMMENT_LINE_RE = re.compile(r"//.*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def normalize_schema(row: dict[str, Any], source: str, idx: int) -> dict[str, Any]:
    instruction = row.get("instruction") or row.get("prompt") or row.get("description") or ""
    code = row.get("code") or row.get("output") or row.get("response") or row.get("solution") or ""
    return {
        "id": f"{source}_{idx:05d}",
        "source": source,
        "instruction": instruction.strip(),
        "code": code.strip(),
        "tags": {},
        "split": "train",
    }


def canonicalise_for_hashing(code: str) -> str:
    """Strip comments, collapse whitespace, canonicalise identifiers --
    the transform MinHash dedup runs over, per Part 3 step 2. Canonicalising
    identifiers means two modules that differ only by variable naming still
    hash as near-duplicates."""
    code = _COMMENT_BLOCK_RE.sub(" ", code)
    code = _COMMENT_LINE_RE.sub(" ", code)

    counter = {"n": 0}
    seen: dict[str, str] = {}
    keywords = {
        "module", "endmodule", "input", "output", "inout", "wire", "reg",
        "always", "assign", "begin", "end", "if", "else", "case", "endcase",
        "default", "posedge", "negedge", "parameter", "localparam", "generate",
        "endgenerate", "for", "while", "function", "endfunction",
    }

    def repl(m: re.Match) -> str:
        ident = m.group(0)
        if ident in keywords:
            return ident
        if ident not in seen:
            counter["n"] += 1
            seen[ident] = f"id{counter['n']}"
        return seen[ident]

    code = _IDENT_RE.sub(repl, code)
    code = _WHITESPACE_RE.sub(" ", code).strip()
    return code


def shingles(text: str, k: int = 5) -> set[str]:
    tokens = text.split(" ")
    return {" ".join(tokens[i:i + k]) for i in range(max(1, len(tokens) - k + 1))}


def _minhash(shingle_set: set[str], num_perm: int = 128):
    from datasketch import MinHash
    mh = MinHash(num_perm=num_perm)
    for s in shingle_set:
        mh.update(s.encode("utf-8"))
    return mh


def dedup_near_duplicates(rows: list[dict[str, Any]], threshold: float = 0.85) -> tuple[list[dict[str, Any]], int]:
    try:
        from datasketch import MinHashLSH
    except ImportError:
        return _dedup_exact_hash(rows)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        canon = canonicalise_for_hashing(row["code"])
        mh = _minhash(shingles(canon))
        key = row["id"]
        if lsh.query(mh):
            dropped += 1
            continue
        lsh.insert(key, mh)
        kept.append(row)
    return kept, dropped


def _dedup_exact_hash(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Fallback if datasketch isn't installed: exact-hash only. Logged
    loudly because it under-catches near-duplicates -- see Part 3 step 2."""
    print("[build_corpus] WARNING: datasketch not installed, falling back to "
          "exact-hash dedup only. Install datasketch for real near-dup removal.")
    seen: set[str] = set()
    kept, dropped = [], 0
    for row in rows:
        h = hashlib.sha256(canonicalise_for_hashing(row["code"]).encode("utf-8")).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(row)
    return kept, dropped


def drop_eval_contamination(
    rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from datasketch import MinHashLSH
    except ImportError:
        print("[build_corpus] WARNING: datasketch not installed, skipping "
              "contamination check. This is a correctness risk -- install "
              "datasketch before trusting pass@1 numbers.")
        return rows, 0

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, erow in enumerate(eval_rows):
        canon = canonicalise_for_hashing(erow.get("code", "") or erow.get("ref_solution", ""))
        if not canon:
            continue
        lsh.insert(f"eval_{i}", _minhash(shingles(canon)))

    kept, dropped = [], 0
    for row in rows:
        canon = canonicalise_for_hashing(row["code"])
        mh = _minhash(shingles(canon))
        if lsh.query(mh):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def build(sources: dict[str, str], eval_paths: list[str], out_path: str, probe_frac: float, seed: int) -> None:
    all_rows: list[dict[str, Any]] = []
    for source, path in sources.items():
        raw_rows = list(read_jsonl(path))
        for idx, raw in enumerate(raw_rows):
            row = normalize_schema(raw, source, idx)
            if row["instruction"] and row["code"]:
                all_rows.append(row)
        print(f"[build_corpus] {source}: {len(raw_rows)} raw -> {len(all_rows)} cumulative valid rows")

    all_rows, n_near_dup = dedup_near_duplicates(all_rows)
    print(f"[build_corpus] near-dup removal: dropped {n_near_dup}, kept {len(all_rows)}")

    if eval_paths:
        eval_rows: list[dict[str, Any]] = []
        for p in eval_paths:
            eval_rows.extend(read_jsonl(p))
        all_rows, n_contaminated = drop_eval_contamination(all_rows, eval_rows)
        print(f"[build_corpus] eval-contamination removal: dropped {n_contaminated}, kept {len(all_rows)}")

    n_ast, n_regex = 0, 0
    for row in all_rows:
        tags = tag_example(row["code"])
        row["tags"] = {"tier": tags.tier, "constructs": tags.constructs}
        if tags.tagger_backend == "pyverilog":
            n_ast += 1
        else:
            n_regex += 1
    print(f"[build_corpus] tagging backend: {n_ast} via pyverilog AST, {n_regex} via regex fallback "
          f"({n_regex / max(1, len(all_rows)):.1%} fallback rate)")

    all_rows = assign_splits(all_rows, probe_frac=probe_frac, seed=seed)
    n_train = sum(1 for r in all_rows if r["split"] == "train")
    n_probe = sum(1 for r in all_rows if r["split"] == "probe")
    print(f"[build_corpus] split: {n_train} train / {n_probe} probe")

    write_jsonl(out_path, all_rows)
    print(f"[build_corpus] wrote {len(all_rows)} rows -> {out_path}")


def _parse_sources(pairs: Iterable[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", required=True, help="name=path.jsonl pairs")
    ap.add_argument("--eval-sets", nargs="*", default=[], help="eval jsonl paths to check contamination against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    build(_parse_sources(args.sources), args.eval_sets, args.out, args.probe_frac, args.seed)


if __name__ == "__main__":
    main()
