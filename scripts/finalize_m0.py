"""Finalize a run early at a specific checkpoint, when the full planned
training budget won't be reached (here: persistent T4 fp16 instability
made continuing past step 600 not worth further GPU-hours and debugging
time -- see run_meta.json's stopped_early_reason for the full context).

Writes artifacts/<run>/final/ + run_meta.json in the exact format sft.py
itself writes on natural completion, so everything downstream --
M1's assert_matches_m0 (checkpoint hash + step count), the diagnostic
pass, eval -- works unmodified against this as if it were a normal
finished run, just with fewer total_steps than originally planned.

    python scripts/finalize_m0.py --checkpoint artifacts/m0/checkpoint-600 \
        --config configs/m0_baseline_kaggle_recovery.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.train.sft import attach_lora, checkpoint_hash, load_base_model_and_tokenizer, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="e.g. artifacts/m0/checkpoint-600")
    ap.add_argument("--config", required=True, help="the config this checkpoint was trained under")
    ap.add_argument(
        "--reason", default=None,
        help="why this run was stopped early; defaults to the T4 fp16 instability explanation",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint)
    cfg = load_config(args.config)
    run_dir = ckpt_dir.parent
    final_dir = run_dir / "final"

    import torch
    trainer_state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
    actual_steps = trainer_state["step"] + 1  # steps 0..step were completed
    elapsed_hours = trainer_state.get("elapsed_hours", 0.0)

    try:
        gpu_name = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu_name = "unknown"

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg, resume_adapter_path=ckpt_dir)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    default_reason = (
        "Persistent fp16 numerical instability on T4 (LoRA weight/gradient "
        "overflow recurring every ~150-200 steps despite gradient clipping "
        "and per-example NaN filtering) made continuing to the full planned "
        "2997 steps not worth the additional GPU-hours and debugging time. "
        "Stopped at the last checkpoint independently verified to have "
        "finite, small-magnitude adapter weights (max |w| = 0.094)."
    )

    meta = {
        "run_name": cfg["run_name"],
        "total_steps": actual_steps,
        "checkpoint_hash": checkpoint_hash(model),
        "gpu": gpu_name,
        "train_gpu_hours": elapsed_hours,
        "config": cfg,
        "realised_histogram": {},
        "stopped_early": True,
        "stopped_early_reason": args.reason or default_reason,
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[finalize] wrote {final_dir} and {run_dir / 'run_meta.json'}")
    print(f"[finalize] total_steps={actual_steps} (originally planned 2997) -- "
          f"M1/diagnostics/eval will now treat this as {cfg['run_name']}'s complete run")


if __name__ == "__main__":
    main()
