"""The bounded online repair loop (Part 9).

`generate_with_repair` takes a plain callable `model_generate_fn(prompt) ->
str` rather than a model object directly, so the loop logic is unit-
testable without loading a multi-GB checkpoint (see tests/test_repair.py).
The CLI entry point below wires it up to a real adapter via
infer/generate.py's loader.

Design points preserved from the guide, each with a reason a reviewer
would ask about:
  - raw, unmodified error text goes into the repair prompt -- summarising
    it throws away the line numbers / identifier names that make the loop
    work at all.
  - only the immediately previous attempt is shown, not the full history --
    accumulating failed attempts fills context and empirically biases the
    model toward repeating them.
  - K is a hard bound, and the attempt index at which each success occurs
    is logged as a result, not just plumbing.
  - compile-stage and simulate-stage failures take different feedback
    payloads (compiler stderr vs. testbench mismatch summary). This
    project's harness adds two more pre-simulation stages (lint,
    synthesize); those get the same "raw tool text" treatment as compile,
    since from the model's point of view they're all "a tool rejected
    your code, here's exactly why."
  - the loop is verification-gated: the harness decides pass/fail, never
    the model itself.
  - repair-induced regressions (attempt n compiled, attempt n+1 didn't)
    are tracked explicitly. When `return_policy="best"`, the loop returns
    the highest-progress attempt seen (pass > synthesize-clean >
    lint-clean > compile-clean > nothing) instead of always the last one --
    see Part 13's row on this exact failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.infer.postprocess import finalize
from src.utils.prompts import build_prompt, build_repair_prompt
from src.verify.harness import VerifyResult, verify

_STAGE_RANK = {"compile": 0, "lint": 1, "synthesize": 2, "simulate": 3, "pass": 4}


@dataclass
class RepairTrace:
    attempts: list[dict[str, Any]] = field(default_factory=list)
    regressions: int = 0


def _progress_rank(result: VerifyResult) -> int:
    base = _STAGE_RANK.get(result.stage, 0)
    return base + (1 if result.ok else 0)


def generate_with_repair(
    model_generate_fn: Callable[[str], str],
    spec: str,
    testbench: str,
    top_module: str = "top",
    K: int = 3,
    return_policy: str = "last",
    require_lint_clean: bool = False,
    require_synthesizable: bool = True,
) -> tuple[str, bool, RepairTrace]:
    trace = RepairTrace()
    code = model_generate_fn(build_prompt(spec))

    best_code, best_rank, best_result = code, -1, None

    for attempt in range(K + 1):
        result = verify(
            code, testbench, top_module=top_module,
            require_lint_clean=require_lint_clean,
            require_synthesizable=require_synthesizable,
        )
        rank = _progress_rank(result)

        if attempt > 0 and trace.attempts:
            prev_rank = trace.attempts[-1]["progress_rank"]
            if rank < prev_rank:
                trace.regressions += 1

        trace.attempts.append({
            "attempt": attempt, "stage": result.stage, "label": result.error_label,
            "ok": result.ok, "progress_rank": rank,
        })

        if rank > best_rank:
            best_code, best_rank, best_result = code, rank, result

        if result.ok:
            final_code = finalize(code)
            return final_code, True, trace

        if attempt == K:
            break

        code = model_generate_fn(
            build_repair_prompt(spec, code, result.error_text, result.stage)
        )

    returned = code if return_policy == "last" else best_code
    return finalize(returned), False, trace


def make_hf_generate_fn(model, tokenizer, temperature: float = 0.2, top_p: float = 0.95,
                         max_new_tokens: int = 768) -> Callable[[str], str]:
    """Greedy-ish single-sample generation, used both for the initial
    attempt and every repair turn. Low temperature by default: the repair
    loop wants the model's best correction given explicit error feedback,
    not exploration."""
    import torch

    def _gen(prompt: str) -> str:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5), top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return _gen
