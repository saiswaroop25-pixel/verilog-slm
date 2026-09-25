"""Regression coverage for the eval-set row schema: tier lives at
row["tags"]["tier"] (same nesting as the corpus schema diagnose.py reads),
not a flat row["tier"] -- a mismatch that went uncaught until the first
real Kaggle eval run, since nothing in this suite previously exercised
run_no_repair_cell/run_repair_cell against a real eval-set-shaped row.
Fakes out generation/verification (GPU + iverilog) the same way
test_repair.py does, per that file's own module docstring rationale."""

from dataclasses import dataclass

import src.eval.run_eval as run_eval_mod
import src.infer.repair as repair_mod


@dataclass
class FakeVerifyResult:
    stage: str
    ok: bool
    error_text: str = "boom"
    error_label: str = "none"


class FakePool:
    """Stands in for ProcessPoolExecutor -- runs map() synchronously in
    this process instead of spawning workers, so a monkeypatched
    _verify_one is actually seen (multiprocessing.spawn, the Windows
    default, re-imports modules fresh in child processes and would not
    inherit this test's monkeypatch)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, args):
        return [fn(a) for a in args]


def _eval_row(tier="T1"):
    return {
        "id": "p1", "instruction": "spec", "testbench": "tb",
        "top_module": "top", "tags": {"tier": tier},
    }


def test_run_no_repair_cell_reads_nested_tier(monkeypatch):
    monkeypatch.setattr(
        run_eval_mod, "generate_batch",
        lambda model, tok, prompts, **kw: [["module m; endmodule"] for _ in prompts],
    )
    monkeypatch.setattr(run_eval_mod, "ProcessPoolExecutor", lambda: FakePool())
    monkeypatch.setattr(
        run_eval_mod, "_verify_one",
        lambda args: {"stage": "pass", "ok": True, "error_label": "none", "tier": args[3]},
    )

    results = run_eval_mod.run_no_repair_cell(model=None, tokenizer=None, eval_rows=[_eval_row("T3")], n=1)
    assert len(results) == 1
    assert results[0]["tier"] == "T3"
    assert results[0]["problem_id"] == "p1"


def test_run_repair_cell_reads_nested_tier(monkeypatch):
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: FakeVerifyResult("pass", True))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)
    monkeypatch.setattr(run_eval_mod, "make_hf_generate_fn", lambda model, tok, **kw: (lambda prompt: "code"))

    results, traces = run_eval_mod.run_repair_cell(model=None, tokenizer=None, eval_rows=[_eval_row("T4")], k_repair=3)
    assert len(results) == 1
    assert results[0]["tier"] == "T4"
    assert results[0]["problem_id"] == "p1"
    assert results[0]["ok"] is True
