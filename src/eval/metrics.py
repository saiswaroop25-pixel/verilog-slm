"""All metrics from Part 10, plus the bootstrap CI and per-category table
that make the results defensible rather than just numbers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021 / Codex paper). Naive
    pass@k (generate exactly k, check any) has high variance and is not
    comparable to published numbers -- every paper cited in this project
    uses this form."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_pass_at_k_per_problem(results: list[list[bool]], ks: list[int] = [1, 5]) -> dict[int, list[float]]:
    """results[i] = list of n bool outcomes for problem i. Returns, per k,
    the list of per-problem pass@k values (before averaging) so a bootstrap
    CI can resample over problems."""
    out: dict[int, list[float]] = {k: [] for k in ks}
    for outcomes in results:
        n = len(outcomes)
        c = sum(outcomes)
        for k in ks:
            out[k].append(pass_at_k(n, c, k))
    return out


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 1337) -> tuple[float, float, float]:
    """Bootstrap over problems (not over samples-within-problem). Returns
    (mean, lo, hi) for a (1-alpha) CI. With 206 eval problems, small
    point-differences need this -- see Part 10 point 6."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def syntax_error_rate(verify_results: list[dict[str, Any]], stage_filter: str | None = None) -> float:
    """compile_failures / total_generations. Pass stage_filter='compile'
    to restrict to first-attempt-only records if the caller pre-filters,
    otherwise this counts any record whose *first* recorded stage is
    'compile' and not ok."""
    if not verify_results:
        return 0.0
    n_fail = sum(1 for r in verify_results if r["stage"] == "compile" and not r["ok"])
    return n_fail / len(verify_results)


def per_category_error_rate(verify_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """error_rate[label] = count / total, for every non-'none' label seen."""
    total = len(verify_results)
    counts: dict[str, int] = defaultdict(int)
    for r in verify_results:
        if not r["ok"] and r["error_label"] != "none":
            counts[r["error_label"]] += 1
    return {label: {"count": c, "rate": c / total if total else 0.0} for label, c in counts.items()}


def per_category_delta_table(m0_results: list[dict[str, Any]], m1_results: list[dict[str, Any]],
                              targeted_labels: set[str]) -> list[dict[str, Any]]:
    m0_rates = per_category_error_rate(m0_results)
    m1_rates = per_category_error_rate(m1_results)
    labels = sorted(set(m0_rates) | set(m1_rates))
    table = []
    for label in labels:
        r0 = m0_rates.get(label, {"rate": 0.0})["rate"]
        r1 = m1_rates.get(label, {"rate": 0.0})["rate"]
        table.append({
            "label": label, "m0": r0, "m1": r1, "delta": r1 - r0,
            "targeted": label in targeted_labels,
        })
    return table


def taxonomy_table(verify_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Part 6's diagnostic table: label, count, share of failures, tier
    concentration. Requires each record to also carry 'tier' (the
    structural tier of the problem it came from)."""
    failures = [r for r in verify_results if not r["ok"]]
    total_failures = len(failures)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in failures:
        by_label[r["error_label"]].append(r)

    table = []
    for label, rows in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        tier_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            tier_counts[r.get("tier", "unknown")] += 1
        tier_concentration = {
            t: c / len(rows) for t, c in sorted(tier_counts.items(), key=lambda kv: -kv[1])
        }
        table.append({
            "label": label, "count": len(rows),
            "share_of_failures": len(rows) / total_failures if total_failures else 0.0,
            "tier_concentration": tier_concentration,
        })
    return table


def construct_failure_rate_table(verify_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-construct-tag failure rate: of all generations whose corpus row
    carries a given construct tag (e.g. 'case_stmt', 'async_reset' --
    src/verify/taxonomy.py's CONSTRUCT_TAGS), what share failed
    verify_static()/verify()? Requires each record to also carry
    'constructs' (the corpus row's construct tag list).

    This -- not taxonomy_table()'s error-LABEL breakdown -- is what
    train/curriculum.py's compute_reweighting() needs: a score keyed by
    construct tag, since the curriculum oversamples examples by which
    constructs they exercise, not by which way generations on them failed.
    Error labels (e.g. 'syntax_error') and construct tags are disjoint
    name spaces; feeding taxonomy_table()'s labels into compute_reweighting
    silently produces uniform weights, since none of its lookups ever hit.
    """
    totals: dict[str, int] = defaultdict(int)
    fails: dict[str, int] = defaultdict(int)
    for r in verify_results:
        for tag in r.get("constructs", []):
            totals[tag] += 1
            if not r["ok"]:
                fails[tag] += 1
    return {
        tag: {"count": totals[tag], "fail_rate": fails[tag] / totals[tag]}
        for tag in sorted(totals, key=lambda t: -fails[t] / totals[t])
    }


def catch_all_share(table: list[dict[str, Any]], catch_all_labels: set[str]) -> float:
    total = sum(row["count"] for row in table)
    catch_all = sum(row["count"] for row in table if row["label"] in catch_all_labels)
    return catch_all / total if total else 0.0


def repair_diagnostics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolution-attempt distribution, per-initial-label repair success
    rate, mean extra forward passes, regression rate (Part 10 point 4)."""
    resolution_dist: dict[str, int] = defaultdict(int)
    by_initial_label: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "resolved": 0})
    total_extra_passes = 0
    total_regressions = 0

    for t in traces:
        attempts = t["attempts"]
        ok = attempts[-1]["ok"]
        n_attempts = len(attempts)
        resolution_dist[str(n_attempts) if ok else "unresolved"] += 1
        initial_label = attempts[0]["label"]
        by_initial_label[initial_label]["total"] += 1
        if ok:
            by_initial_label[initial_label]["resolved"] += 1
        total_extra_passes += max(0, n_attempts - 1)
        total_regressions += t.get("regressions", 0)

    n = len(traces) or 1
    return {
        "resolution_distribution": dict(resolution_dist),
        "repair_success_rate_by_initial_label": {
            label: v["resolved"] / v["total"] for label, v in by_initial_label.items()
        },
        "mean_extra_forward_passes": total_extra_passes / n,
        "regression_rate": total_regressions / n,
    }


def accuracy_per_gpu_hour(pass_at_1: float, train_gpu_hours: float) -> float | None:
    if train_gpu_hours <= 0:
        return None
    return pass_at_1 / train_gpu_hours
