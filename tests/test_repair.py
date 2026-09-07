"""Unit-tests the repair-loop *logic* (bounded K, trace shape, regression
detection, return policy) against a fake verifier, so it doesn't need
iverilog/yosys installed. src/verify/harness.py itself gets its own
tool-gated tests in test_harness.py."""

from dataclasses import dataclass

import pytest

import src.infer.repair as repair_mod
from src.infer.repair import generate_with_repair


@dataclass
class FakeResult:
    stage: str
    ok: bool
    error_text: str = "boom"
    error_label: str = "syntax_error"


def test_succeeds_on_first_try(monkeypatch):
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: FakeResult("pass", True))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)

    code, ok, trace = generate_with_repair(lambda prompt: "module m; endmodule", "spec", "tb", K=3)
    assert ok is True
    assert len(trace.attempts) == 1
    assert trace.attempts[0]["attempt"] == 0


def test_bounded_by_k(monkeypatch):
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: FakeResult("compile", False))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)

    calls = {"n": 0}

    def gen(prompt):
        calls["n"] += 1
        return f"attempt {calls['n']}"

    code, ok, trace = generate_with_repair(gen, "spec", "tb", K=3)
    assert ok is False
    # K=3 repairs => 4 total verify attempts (initial + 3 repairs)
    assert len(trace.attempts) == 4
    assert calls["n"] == 4  # initial generation + 3 repair generations


def test_success_on_second_attempt_records_attempt_index(monkeypatch):
    outcomes = [FakeResult("compile", False), FakeResult("pass", True)]
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: outcomes.pop(0))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)

    code, ok, trace = generate_with_repair(lambda p: "code", "spec", "tb", K=3)
    assert ok is True
    assert len(trace.attempts) == 2
    assert trace.attempts[-1]["attempt"] == 1


def test_regression_is_counted(monkeypatch):
    # compile-clean, then simulate-fail (regression: compile > nothing,
    # but this is stage progress compile(0)+ok(0)=... use ranks: compile
    # fail=0, then a *worse* outcome is impossible below compile, so
    # simulate progressive stages: first attempt reaches simulate (rank 3),
    # second attempt regresses to compile failure (rank 0).
    outcomes = [
        FakeResult("simulate", False),
        FakeResult("compile", False),
        FakeResult("compile", False),
        FakeResult("compile", False),
    ]
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: outcomes.pop(0))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)

    code, ok, trace = generate_with_repair(lambda p: "code", "spec", "tb", K=3)
    assert trace.regressions >= 1


def test_return_policy_best_returns_highest_progress_attempt(monkeypatch):
    codes_seen = []

    def gen(prompt):
        codes_seen.append(prompt)
        return f"code_{len(codes_seen)}"

    outcomes = [
        FakeResult("simulate", False),  # attempt 0: good progress
        FakeResult("compile", False),   # attempt 1: regression
        FakeResult("compile", False),
        FakeResult("compile", False),
    ]
    monkeypatch.setattr(repair_mod, "verify", lambda *a, **kw: outcomes.pop(0))
    monkeypatch.setattr(repair_mod, "finalize", lambda code: code)

    code, ok, trace = generate_with_repair(gen, "spec", "tb", K=3, return_policy="best")
    assert ok is False
    assert code == "code_1"  # the attempt-0 output, which reached 'simulate'
