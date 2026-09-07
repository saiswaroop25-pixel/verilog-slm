"""train/probe split (Part 3, step 4). 90/10, stratified by structural
tier so the probe set (what M0 gets diagnosed on) isn't accidentally all
combinational. Eval sets are external and never touched here."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def stratified_split(
    rows: list[dict[str, Any]],
    probe_frac: float = 0.10,
    seed: int = 1337,
    tier_key: str = "tier",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[row["tags"][tier_key]].append(row)

    train, probe = [], []
    for tier, tier_rows in by_tier.items():
        tier_rows = tier_rows[:]
        rng.shuffle(tier_rows)
        n_probe = max(1, round(len(tier_rows) * probe_frac)) if tier_rows else 0
        probe.extend(tier_rows[:n_probe])
        train.extend(tier_rows[n_probe:])

    rng.shuffle(train)
    rng.shuffle(probe)
    return train, probe


def assign_splits(rows: list[dict[str, Any]], probe_frac: float = 0.10, seed: int = 1337) -> list[dict[str, Any]]:
    train, probe = stratified_split(rows, probe_frac=probe_frac, seed=seed)
    train_ids = {r["id"] for r in train}
    for row in rows:
        row["split"] = "train" if row["id"] in train_ids else "probe"
    return rows
