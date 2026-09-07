"""Curriculum builder (Part 7). Two composed mechanisms:

  1. Structural ordering: three phases across the fixed step budget S,
     each phase drawing from a different tier mixture.
  2. Error-driven reweighting: per-example sampling weight derived from
     the M0 diagnostic table (Part 6), so examples touching high-failure
     construct tags get oversampled.

Critically, the step count never changes -- only which examples fill
those steps. That's what keeps M0 vs M1 a one-variable ablation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

DEFAULT_PHASES = [
    {"frac": 0.3, "tier_mix": {"T1": 0.70, "T2": 0.30}},
    {"frac": 0.4, "tier_mix": {"T1": 0.25, "T2": 0.40, "T3": 0.35}},
    {"frac": 0.3, "tier_mix": {"T1": 0.15, "T2": 0.25, "T3": 0.35, "T4": 0.25}},
]


@dataclass
class Phase:
    start_step: int
    end_step: int
    tier_mix: dict[str, float]


def build_phases(total_steps: int, phases: list[dict] = None) -> list[Phase]:
    phases = phases or DEFAULT_PHASES
    out, cursor = [], 0
    for i, p in enumerate(phases):
        length = round(total_steps * p["frac"])
        end = total_steps if i == len(phases) - 1 else cursor + length
        out.append(Phase(start_step=cursor, end_step=end, tier_mix=p["tier_mix"]))
        cursor = end
    return out


def phase_for_step(step: int, phases: list[Phase]) -> Phase:
    for p in phases:
        if p.start_step <= step < p.end_step:
            return p
    return phases[-1]


def compute_reweighting(
    rows: list[dict[str, Any]],
    failure_share: dict[str, float],
    alpha: float | None = None,
    target_max_weight: float = 3.0,
    weight_clip: tuple[float, float] = (1.0, 3.0),
) -> dict[str, float]:
    """w(x) = 1 + alpha * sum(failure_share[tag] for tag in x.construct_tags),
    clipped to weight_clip. alpha is auto-tuned (if not given) so the
    single highest raw score lands at target_max_weight, per Part 7.
    """
    raw_scores: dict[str, float] = {}
    for row in rows:
        tags = row["tags"].get("constructs", [])
        score = sum(failure_share.get(t, 0.0) for t in tags)
        raw_scores[row["id"]] = score

    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    if alpha is None:
        alpha = (target_max_weight - 1.0) / max_raw if max_raw > 0 else 0.0

    weights: dict[str, float] = {}
    lo, hi = weight_clip
    for row_id, score in raw_scores.items():
        w = 1.0 + alpha * score
        weights[row_id] = min(hi, max(lo, w))
    return weights


class PhaseWeightedSampler:
    """A step-indexed sampler: for a given global training step, returns
    one example index drawn from the tier mixture of the active phase,
    weighted within-tier by the error-driven weight. Implemented as a
    plain Python generator so it's framework-agnostic; wrap it in a
    torch IterableDataset / Sampler at the training-loop boundary
    (see src/train/sft.py).
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        total_steps: int,
        weights: dict[str, float] | None = None,
        phases: list[dict] | None = None,
        seed: int = 1337,
    ):
        import random

        self.rows = rows
        self.rng = random.Random(seed)
        self.phases = build_phases(total_steps, phases)
        self.weights = weights or {r["id"]: 1.0 for r in rows}

        self.by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            self.by_tier[r["tags"]["tier"]].append(r)

        self._tier_weighted_pools = {
            tier: (tier_rows, [self.weights.get(r["id"], 1.0) for r in tier_rows])
            for tier, tier_rows in self.by_tier.items()
        }

        self.realised_tier_counts: dict[str, int] = defaultdict(int)
        self.realised_construct_counts: dict[str, int] = defaultdict(int)

    def _sample_tier(self, tier_mix: dict[str, float]) -> str:
        available = {t: p for t, p in tier_mix.items() if self.by_tier.get(t)}
        if not available:
            available = {t: 1.0 for t in self.by_tier}
        total = sum(available.values())
        r = self.rng.random() * total
        acc = 0.0
        for tier, p in available.items():
            acc += p
            if r <= acc:
                return tier
        return next(iter(available))

    def sample(self, step: int) -> dict[str, Any]:
        phase = phase_for_step(step, self.phases)
        tier = self._sample_tier(phase.tier_mix)
        rows, weights = self._tier_weighted_pools[tier]
        chosen = self.rng.choices(rows, weights=weights, k=1)[0]

        self.realised_tier_counts[tier] += 1
        for tag in chosen["tags"].get("constructs", []):
            self.realised_construct_counts[tag] += 1
        return chosen

    def realised_histogram(self) -> dict[str, dict[str, int]]:
        return {
            "tier": dict(self.realised_tier_counts),
            "construct": dict(self.realised_construct_counts),
        }
