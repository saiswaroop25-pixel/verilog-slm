import numpy as np
import pytest

from src.eval.metrics import (
    accuracy_per_gpu_hour,
    bootstrap_ci,
    catch_all_share,
    compute_pass_at_k_per_problem,
    pass_at_k,
    per_category_delta_table,
    repair_diagnostics,
    syntax_error_rate,
    taxonomy_table,
)


def test_pass_at_k_all_pass():
    assert pass_at_k(n=20, c=20, k=1) == 1.0
    assert pass_at_k(n=20, c=20, k=5) == 1.0


def test_pass_at_k_none_pass():
    assert pass_at_k(n=20, c=0, k=1) == 0.0


def test_pass_at_k_matches_naive_at_k1():
    # at k=1, unbiased pass@1 == c/n
    assert abs(pass_at_k(n=20, c=6, k=1) - 6 / 20) < 1e-9


def test_compute_pass_at_k_per_problem():
    results = [[True, False, False], [False, False, False], [True, True, True]]
    out = compute_pass_at_k_per_problem(results, ks=[1])
    assert len(out[1]) == 3
    assert out[1][2] == 1.0  # all-pass problem -> pass@1 = 1.0
    assert out[1][1] == 0.0  # all-fail problem -> pass@1 = 0.0


def test_bootstrap_ci_bounds():
    mean, lo, hi = bootstrap_ci([1.0] * 50, n_boot=200)
    assert mean == 1.0
    assert lo == 1.0 and hi == 1.0


def test_syntax_error_rate():
    results = [
        {"stage": "compile", "ok": False, "error_label": "syntax_error"},
        {"stage": "pass", "ok": True, "error_label": "none"},
    ]
    assert syntax_error_rate(results) == 0.5


def test_taxonomy_table_and_catch_all_share():
    results = [
        {"ok": False, "error_label": "wrong_logic_other", "tier": "T1"},
        {"ok": False, "error_label": "wrong_logic_other", "tier": "T2"},
        {"ok": False, "error_label": "missing_default_case", "tier": "T3"},
        {"ok": True, "error_label": "none", "tier": "T1"},
    ]
    table = taxonomy_table(results)
    assert table[0]["label"] == "wrong_logic_other"
    assert table[0]["count"] == 2
    from src.verify.taxonomy import CATCH_ALL_LABELS
    share = catch_all_share(table, CATCH_ALL_LABELS)
    assert abs(share - 2 / 3) < 1e-9


def test_per_category_delta_table():
    m0 = [{"ok": False, "error_label": "incomplete_sensitivity"}] * 4 + [{"ok": True, "error_label": "none"}] * 6
    m1 = [{"ok": False, "error_label": "incomplete_sensitivity"}] * 1 + [{"ok": True, "error_label": "none"}] * 9
    table = per_category_delta_table(m0, m1, targeted_labels={"incomplete_sensitivity"})
    row = next(r for r in table if r["label"] == "incomplete_sensitivity")
    assert row["m0"] == 0.4
    assert row["m1"] == 0.1
    assert row["delta"] == pytest.approx(-0.3)
    assert row["targeted"] is True


def test_repair_diagnostics():
    traces = [
        {"attempts": [{"ok": False, "label": "syntax_error"}, {"ok": True, "label": "none"}], "regressions": 0},
        {"attempts": [{"ok": False, "label": "syntax_error"}] * 4, "regressions": 1},
    ]
    diag = repair_diagnostics(traces)
    assert diag["resolution_distribution"]["2"] == 1
    assert diag["resolution_distribution"]["unresolved"] == 1
    assert diag["mean_extra_forward_passes"] == (1 + 3) / 2
    assert diag["regression_rate"] == 0.5


def test_accuracy_per_gpu_hour():
    assert accuracy_per_gpu_hour(0.4, 4.0) == 0.1
    assert accuracy_per_gpu_hour(0.4, 0.0) is None
