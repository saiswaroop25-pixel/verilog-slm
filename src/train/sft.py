"""Single SFT entry point, config-driven (Part 1 + Part 5 + Part 8).

    python -m src.train.sft --config configs/m0_baseline.yaml
    python -m src.train.sft --config configs/m1_curriculum.yaml

M0 and M1 must differ in exactly one thing: the sampler (flat-shuffled vs
PhaseWeightedSampler). Everything else -- base checkpoint, optimizer, LR
schedule, total optimizer steps, seed, sequence length -- is read from the
same `base.yaml` block both configs include. This file enforces that by
asserting the resolved base-checkpoint hash and max_steps match whatever
M0 recorded, whenever `sampler.type: curriculum` and an `assert_against`
path is given (see configs/m1_curriculum.yaml).

Resumable: every `save_steps` interval (and at completion) writes the LoRA
adapter plus a `trainer_state.pt` (optimizer/scheduler/step/RNG state) into
`checkpoint-N/`. Re-running the same command against a `run_name` that
already has checkpoints picks up from the latest one instead of restarting
-- required because a real M0 run (~36 GPU-hours on a T4) outlives a single
Colab session. If `./artifacts_drive_ckpt` exists (the Drive-backed symlink
notebooks/02_train.ipynb's bootstrap cell creates), every checkpoint is
also mirrored there and synced back on startup, since local `artifacts/`
lives on the Colab VM's ephemeral disk and does not survive a disconnect.

Requires: torch, transformers, peft, bitsandbytes, accelerate. These are
GPU-flow dependencies (Part 0/2 of the guide) and are intentionally not
imported at module load time so the rest of the repo (data pipeline,
verify harness, eval metrics) stays runnable on a CPU-only machine without
installing a 5GB CUDA stack -- see requirements.txt for the split between
core and train extras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from src.utils.io_utils import append_jsonl, read_jsonl, write_jsonl
from src.utils.prompts import format_sft_example
from src.utils.seeding import set_seed


def load_config(path: str) -> dict[str, Any]:
    """Configs are layered: a config can set `extends: base.yaml`. The
    child overrides win, everything else is inherited."""
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "extends" in cfg:
        base_path = cfg_path.parent / cfg.pop("extends")
        base_cfg = load_config(str(base_path))
        merged = _deep_merge(base_cfg, cfg)
        return merged
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_run_dir(cfg: dict[str, Any]) -> Path:
    run_dir = Path(cfg["output"]["artifacts_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
    candidates = []
    for p in run_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    step, path = max(candidates, key=lambda x: x[0])
    return path, step


def drive_mirror_dir(run_dir: Path) -> Path | None:
    """See module docstring. Returns None (no-op everywhere else in this
    file) when not running in a notebook that set this symlink up, e.g.
    local/CI runs of a debug config."""
    drive_root = Path("artifacts_drive_ckpt")
    if not drive_root.exists():
        return None
    mirror = drive_root / run_dir.name
    mirror.mkdir(parents=True, exist_ok=True)
    return mirror


def sync_from_drive(run_dir: Path, mirror: Path | None) -> None:
    if mirror is None:
        return
    import shutil
    for item in mirror.iterdir():
        dest = run_dir / item.name
        if item.is_dir():
            if not dest.exists():
                shutil.copytree(item, dest)
        elif not dest.exists() or item.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(item, dest)


def sync_to_drive(run_dir: Path, mirror: Path | None, name: str) -> None:
    if mirror is None:
        return
    import shutil
    src = run_dir / name
    dest = mirror / name
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)


def rng_state_dict() -> dict[str, Any]:
    import random
    import torch

    state = {"python_random": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state: dict[str, Any]) -> None:
    import random
    import torch

    random.setstate(state["python_random"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    run_dir: Path, tag: str, model, optimizer, scheduler, scaler,
    step: int, running_loss: float, sampler, elapsed_hours: float,
    mirror: Path | None,
) -> None:
    import torch

    ckpt_dir = run_dir / tag
    model.save_pretrained(ckpt_dir)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "running_loss": running_loss,
        "elapsed_hours": elapsed_hours,
        "sampler_rng_state": sampler.rng.getstate() if hasattr(sampler, "rng") else None,
        **rng_state_dict(),
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")
    sync_to_drive(run_dir, mirror, tag)


def checkpoint_hash(model) -> str:
    """A cheap fingerprint of the base checkpoint (config + first-layer
    weight sample), used to assert M0 and M1 started from literally the
    same weights -- catches the "oops I re-downloaded a different
    revision" class of bug that quietly invalidates an ablation."""
    import torch

    h = hashlib.sha256()
    h.update(json.dumps(model.config.to_dict(), sort_keys=True).encode())
    with torch.no_grad():
        first_param = next(model.parameters())
        h.update(first_param.flatten()[:1024].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def load_base_model_and_tokenizer(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = cfg["model"]["name"]
    quant = cfg.get("quantization", {})

    bnb_config = None
    if quant.get("load_in_4bit"):
        compute_dtype = getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float16,
        # Without this, some transformers versions/model classes fall back to
        # the naive "eager" attention path, which materialises the full
        # batch x heads x seq_len x seq_len score matrix in memory instead of
        # using a fused kernel -- at seq_len=2048 that tensor alone is
        # hundreds of MB *per layer*, and was a real contributor to an
        # out-of-memory crash observed on a T4 even with a 0.5B model.
        attn_implementation="sdpa",
    )
    return model, tokenizer


def attach_lora(model, cfg: dict[str, Any], resume_adapter_path: Path | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    # Belt-and-suspenders: prepare_model_for_kbit_training is supposed to
    # wire this up itself (a hook that makes the input embeddings require
    # grad), but when the base model is fully frozen (as it is here -- only
    # the LoRA adapters train), a missing/broken version of that wiring is a
    # well-documented way for gradient checkpointing to silently stop
    # actually checkpointing: autograd needs at least one requires_grad=True
    # tensor flowing into each checkpointed region, or it can end up
    # retaining full activations anyway, defeating the whole point (and
    # inflating memory well beyond what a frozen-base + LoRA setup should
    # need). Calling this explicitly costs nothing if peft already did it.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if resume_adapter_path is not None:
        # Load the trained adapter weights from the checkpoint rather than
        # a fresh random LoRA init -- is_trainable=True or peft defaults to
        # eval-mode adapters (frozen), which would silently turn "resume"
        # into "resume as a no-op inference model".
        model = PeftModel.from_pretrained(model, str(resume_adapter_path), is_trainable=True)
        print(f"[sft] resumed LoRA adapter weights from {resume_adapter_path}")
    else:
        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # Keep the (small) set of trainable LoRA params in fp32 regardless of
    # the frozen 4-bit base's compute dtype. Two multi-hundred-step fp16
    # training runs on T4 both independently drifted into a state where
    # every forward pass produced NaN -- consistent with fp16-precision
    # rounding error accumulating over hundreds of small optimizer updates
    # applied directly to fp16-stored weights, not just isolated gradient
    # spikes (which clipping already guards against separately). This is
    # standard QLoRA practice and costs negligible memory since only the
    # adapter matrices are affected, not the frozen base.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return model


def build_dataset_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(cfg["data"]["corpus_path"]) if r["split"] == "train"]
    if not rows:
        raise ValueError(f"No train-split rows found in {cfg['data']['corpus_path']}")
    return rows


def build_sampler(cfg: dict[str, Any], rows: list[dict[str, Any]], total_steps: int):
    sampler_cfg = cfg["sampler"]
    seed = cfg["training"]["seed"]

    if sampler_cfg["type"] == "flat":
        import random

        class FlatSampler:
            def __init__(self, seed: int):
                self.rng = random.Random(seed)

            def sample(self, step: int):
                return self.rng.choice(rows)

            def realised_histogram(self):
                return {}

        return FlatSampler(seed)

    if sampler_cfg["type"] == "curriculum":
        from src.train.curriculum import PhaseWeightedSampler

        weights = None
        diag_path = sampler_cfg.get("diagnostic_table_path")
        if diag_path and Path(diag_path).exists():
            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)
            failure_share = {row["label"]: row["share_of_failures"] for row in diag["labels"]}
            from src.train.curriculum import compute_reweighting
            weights = compute_reweighting(rows, failure_share)

        return PhaseWeightedSampler(
            rows, total_steps=total_steps, weights=weights,
            phases=sampler_cfg.get("phases"), seed=seed,
        )

    raise ValueError(f"Unknown sampler.type: {sampler_cfg['type']}")


def assert_matches_m0(cfg: dict[str, Any], model, total_steps: int) -> None:
    assert_path = cfg["sampler"].get("assert_against")
    if not assert_path or not Path(assert_path).exists():
        return
    with open(assert_path, "r", encoding="utf-8") as f:
        m0_meta = json.load(f)

    assert total_steps == m0_meta["total_steps"], (
        f"max_steps mismatch: M1={total_steps} vs M0={m0_meta['total_steps']}. "
        "M1 must reuse M0's realised step count exactly."
    )
    this_hash = checkpoint_hash(model)
    assert this_hash == m0_meta["checkpoint_hash"], (
        f"base checkpoint hash mismatch: M1={this_hash} vs M0={m0_meta['checkpoint_hash']}. "
        "M0 and M1 must start from the identical base checkpoint."
    )
    print(f"[sft] assertion passed: max_steps={total_steps}, checkpoint_hash={this_hash} match M0")


def train(cfg: dict[str, Any]) -> None:
    # Cheap, harmless insurance against allocator fragmentation on small
    # GPUs (T4/L4) -- torch's CUDA allocator suggested this itself in an
    # OOM observed during development. Must be set before the first CUDA
    # op, so as early as possible here.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(cfg["training"]["seed"])
    run_dir = resolve_run_dir(cfg)
    log_path = run_dir / "train_log.jsonl"

    mirror = drive_mirror_dir(run_dir)
    sync_from_drive(run_dir, mirror)

    final_dir = run_dir / "final"
    if final_dir.exists() and (run_dir / "run_meta.json").exists():
        print(f"[sft] {run_dir} already has a completed run (final/ + run_meta.json) -- "
              "nothing to resume. Delete the run directory to retrain from scratch.")
        return

    resume = find_latest_checkpoint(run_dir)
    resume_path = resume[0] if resume else None

    import subprocess
    try:
        gpu_name = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu_name = "unknown"
    print(f"[sft] GPU: {gpu_name}")

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg, resume_adapter_path=resume_path)

    rows = build_dataset_rows(cfg)

    total_steps = cfg["training"].get("max_steps")
    if total_steps is None:
        assert_path = cfg.get("sampler", {}).get("assert_against")
        if assert_path and Path(assert_path).exists():
            # M1: reuse M0's realised step count exactly rather than
            # re-deriving from epochs -- with a reweighted sampler, "3
            # epochs" no longer means the same number of steps (Part 5).
            with open(assert_path, "r", encoding="utf-8") as f:
                total_steps = json.load(f)["total_steps"]
            print(f"[sft] max_steps not set; inherited S={total_steps} from {assert_path}")
        else:
            effective_batch = cfg["training"]["per_device_batch_size"] * cfg["training"]["grad_accum_steps"]
            total_steps = (len(rows) * cfg["training"]["epochs"]) // effective_batch

    assert_matches_m0(cfg, model, total_steps)

    sampler = build_sampler(cfg, rows, total_steps)

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Dynamic loss scaling: fp16 (forced on T4 -- no bf16 tensor cores) has
    # a much narrower representable range than fp32, and gradients can
    # silently underflow to zero or overflow to inf without it. This is
    # the standard, purpose-built fix -- rescales the loss before backward
    # so gradients land in fp16's representable range, and automatically
    # backs the scale off (skipping that step's update) whenever an
    # inf/nan is detected, rather than us hand-rolling that detection.
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    seq_len = cfg["training"]["seq_len"]
    per_device_bs = cfg["training"]["per_device_batch_size"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    save_steps = cfg["training"].get("save_steps", 200)
    max_wall_hours = cfg["training"].get("max_wall_hours")
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    nan_skips = 0

    start_step = 0
    running_loss = 0.0
    prior_elapsed_hours = 0.0

    if resume_path is not None:
        import torch
        trainer_state = torch.load(resume_path / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        if "scaler" in trainer_state:
            scaler.load_state_dict(trainer_state["scaler"])
        # load_state_dict overwrites scheduler.base_lrs with whatever the
        # checkpoint recorded, silently undoing a deliberate LR change in
        # this run's config (e.g. lowering it to recover from fp16
        # instability) -- only the schedule's *progress* (warmup/cosine
        # position) should come from the checkpoint, not its target LR.
        scheduler.base_lrs = [cfg["training"]["lr"] for _ in scheduler.base_lrs]
        running_loss = trainer_state["running_loss"]
        if running_loss != running_loss:  # NaN check (NaN is the only value unequal to itself)
            # The EMA (0.98*old + 0.02*new) never resets on resume -- once
            # a single non-finite step_loss poisoned it, every later step
            # multiplies that NaN forward forever, even after the weights
            # and every subsequent step are actually fine (as independently
            # confirmed by checking the saved adapter/optimizer tensors for
            # non-finite values). It's a display statistic only, not used
            # for any training decision, so just reset it.
            print("[sft] resumed running_loss was NaN (an earlier non-finite step poisoned "
                  "the EMA permanently) -- resetting it")
            running_loss = 0.0
        prior_elapsed_hours = trainer_state.get("elapsed_hours", 0.0)
        start_step = trainer_state["step"] + 1
        load_rng_state_dict(trainer_state)
        if trainer_state.get("sampler_rng_state") is not None and hasattr(sampler, "rng"):
            sampler.rng.setstate(trainer_state["sampler_rng_state"])
        print(f"[sft] resuming from step {trainer_state['step']}/{total_steps} "
              f"(checkpoint {resume_path}), prior elapsed {prior_elapsed_hours:.2f} GPU-hours")

    model.train()
    start_time = time.monotonic()

    for step in range(start_step, total_steps):
        micro_losses = []
        for _ in range(grad_accum):
            batch_rows = [sampler.sample(step) for _ in range(per_device_bs)]
            texts = [format_sft_example(r["instruction"], r["code"]) for r in batch_rows]
            enc = tokenizer(
                texts, return_tensors="pt", truncation=True, max_length=seq_len,
                padding=True,
            ).to(model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**enc, labels=labels)
            # Gradients from every micro-batch in this step get summed via
            # repeated .backward() calls before the optimizer step -- a
            # single example whose forward pass produces a non-finite loss
            # (a corpus-data issue, independent of weight magnitude: this
            # reproduced at the identical step across different LRs, since
            # sampler.rng resumes deterministically) would otherwise poison
            # the entire accumulated gradient. Identify and skip just that
            # example's contribution instead -- GradScaler below handles
            # the separate case of a fine loss whose gradient overflows.
            if not torch.isfinite(out.loss):
                bad_ids = [r.get("id", "?") for r in batch_rows]
                print(f"[sft] step {step}: non-finite loss from example(s) {bad_ids} -- "
                      f"excluding from this step's accumulated gradient")
                continue
            loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            micro_losses.append(loss.item() * grad_accum)

        if not micro_losses:
            print(f"[sft] step {step}: every micro-batch produced a non-finite loss -- "
                  f"skipping this step entirely")
            optimizer.zero_grad()
            scheduler.step()
            continue

        scale_before = scaler.get_scale()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            nan_skips += 1
            print(f"[sft] step {step}: GradScaler detected inf/nan (grad norm {float(grad_norm)}), "
                  f"skipped optimizer step and backed off scale to {scaler.get_scale():.0f} "
                  f"({nan_skips} skipped so far)")
        scheduler.step()
        optimizer.zero_grad()

        step_loss = sum(micro_losses) / len(micro_losses)
        running_loss = 0.98 * running_loss + 0.02 * step_loss if step > 0 else step_loss

        if step % 10 == 0:
            print(f"[sft] step {step}/{total_steps} loss={step_loss:.4f} ema={running_loss:.4f}")

        append_jsonl(log_path, {
            "run_id": cfg["run_name"], "step": step, "loss": step_loss,
            "lr": scheduler.get_last_lr()[0], "wall_clock": time.time(),
            "gpu": gpu_name,
        })

        elapsed_hours = prior_elapsed_hours + (time.monotonic() - start_time) / 3600
        # Some platforms (Kaggle: 12h/session hard cap) kill the process
        # outright at a wall-clock limit rather than just disconnecting
        # (Colab) -- if that lands mid-checkpoint-write, the next resume
        # attempt finds a truncated checkpoint-N/. Self-stop with margin
        # instead: save cleanly and exit before the platform does it for us.
        over_wall_budget = max_wall_hours is not None and elapsed_hours >= max_wall_hours

        if over_wall_budget or (step > 0 and step % save_steps == 0):
            save_checkpoint(
                run_dir, f"checkpoint-{step}", model, optimizer, scheduler, scaler,
                step, running_loss, sampler, elapsed_hours, mirror,
            )
            sync_to_drive(run_dir, mirror, "train_log.jsonl")

        if over_wall_budget:
            print(f"[sft] wall-clock budget ({max_wall_hours}h) reached at step "
                  f"{step}/{total_steps} -- exiting cleanly for resume next session")
            return

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    sync_to_drive(run_dir, mirror, "final")

    total_hours = prior_elapsed_hours + (time.monotonic() - start_time) / 3600
    meta = {
        "run_name": cfg["run_name"],
        "total_steps": total_steps,
        "checkpoint_hash": checkpoint_hash(model),
        "gpu": gpu_name,
        "train_gpu_hours": total_hours,
        "config": cfg,
        "realised_histogram": sampler.realised_histogram(),
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    sync_to_drive(run_dir, mirror, "run_meta.json")
    print(f"[sft] done. {total_steps} steps in {total_hours:.2f} GPU-hours. Artifacts -> {run_dir}")


def main() -> None:
    # Piping this through `tee` (as the notebook does, to also persist a
    # log file) makes Python switch stdout from line-buffered (its default
    # when attached to a terminal) to full block buffering -- progress
    # prints can then sit unflushed for hours of real, healthy training
    # before enough output accumulates to appear at all, which reads
    # exactly like a hang. Force line buffering regardless of what stdout
    # is connected to.
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if "run_name" not in cfg:
        cfg["run_name"] = Path(args.config).stem
    train(cfg)


if __name__ == "__main__":
    main()
