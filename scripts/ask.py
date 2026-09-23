"""Interactive single-question inference: give the trained model one
Verilog specification, get back one finished module. Everything else in
src/infer/generate.py is batch-oriented (n samples x many problems, for
the diagnostic pass and eval) -- this is the ad-hoc "ask it a question"
entry point for actually using the trained model.

    python -m scripts.ask --adapter artifacts/m1/final \
        --question "4-bit synchronous up counter with active-low reset"

    echo "8-to-1 mux with 3-bit select" | python -m scripts.ask --adapter artifacts/m0/final

Defaults to artifacts/m1/final if present, else artifacts/m0/final --
the best trained model available, so callers who don't care which run
produced it don't have to know.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.infer.generate import generate_batch, load_model_for_inference
from src.infer.postprocess import finalize
from src.utils.prompts import build_prompt
from src.utils.seeding import set_seed


def default_adapter() -> str:
    if Path("artifacts/m1/final").exists():
        return "artifacts/m1/final"
    return "artifacts/m0/final"


def ask(adapter: str, question: str, temperature: float, top_p: float, seed: int) -> str:
    set_seed(seed)
    model, tokenizer = load_model_for_inference(adapter)
    prompt = build_prompt(question)
    [[completion]] = generate_batch(model, tokenizer, [prompt], n=1, temperature=temperature, top_p=top_p)
    return finalize(completion)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="defaults to artifacts/m1/final, else artifacts/m0/final")
    ap.add_argument("--question", default=None, help="the Verilog spec; reads stdin if omitted")
    ap.add_argument("--temperature", type=float, default=0.2, help="low by default -- one answer, not a sample")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    adapter = args.adapter or default_adapter()
    question = args.question or sys.stdin.read().strip()
    if not question:
        ap.error("no question given: pass --question or pipe one in on stdin")

    print(ask(adapter, question, args.temperature, args.top_p, args.seed))


if __name__ == "__main__":
    main()
