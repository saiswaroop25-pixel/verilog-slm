from src.train.curriculum import (
    PhaseWeightedSampler,
    build_phases,
    compute_reweighting,
    phase_for_step,
)


def _row(id_, tier, constructs=None):
    return {"id": id_, "tags": {"tier": tier, "constructs": constructs or []}}


def test_build_phases_covers_full_range_no_gaps():
    phases = build_phases(1000)
    assert phases[0].start_step == 0
    assert phases[-1].end_step == 1000
    for a, b in zip(phases, phases[1:]):
        assert a.end_step == b.start_step


def test_phase_for_step():
    phases = build_phases(100)
    assert phase_for_step(0, phases) is phases[0]
    assert phase_for_step(99, phases) is phases[-1]


def test_compute_reweighting_clips_and_scales():
    rows = [
        _row("a", "T2", ["incomplete_sensitivity"]),
        _row("b", "T1", []),
    ]
    failure_share = {"incomplete_sensitivity": 1.0}
    weights = compute_reweighting(rows, failure_share, target_max_weight=3.0)
    assert weights["b"] == 1.0
    assert abs(weights["a"] - 3.0) < 1e-6


def test_sampler_only_draws_from_available_tiers():
    rows = [_row(f"t1_{i}", "T1") for i in range(5)]
    sampler = PhaseWeightedSampler(rows, total_steps=50, seed=0)
    for step in range(50):
        chosen = sampler.sample(step)
        assert chosen["tags"]["tier"] == "T1"


def test_sampler_step_count_is_caller_controlled():
    rows = [_row(f"t1_{i}", "T1") for i in range(3)] + [_row(f"t2_{i}", "T2") for i in range(3)]
    sampler = PhaseWeightedSampler(rows, total_steps=200, seed=0)
    n_draws = 0
    for step in range(200):
        sampler.sample(step)
        n_draws += 1
    assert n_draws == 200  # curriculum changes *which* examples, never step count
