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
    )
    return model, tokenizer


def attach_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


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
        rng = random.Random(seed)

        class FlatSampler:
            def sample(self, step: int):
                return rng.choice(rows)

            def realised_histogram(self):
                return {}

        return FlatSampler()

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
    set_seed(cfg["training"]["seed"])
    run_dir = resolve_run_dir(cfg)
    log_path = run_dir / "train_log.jsonl"

    import subprocess
    try:
        gpu_name = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu_name = "unknown"
    print(f"[sft] GPU: {gpu_name}")

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

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

    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    seq_len = cfg["training"]["seq_len"]
    per_device_bs = cfg["training"]["per_device_batch_size"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    save_steps = cfg["training"].get("save_steps", 200)

    model.train()
    start_time = time.monotonic()
    running_loss = 0.0

    for step in range(total_steps):
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
            out = model(**enc, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            micro_losses.append(loss.item() * grad_accum)

        optimizer.step()
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

        if step > 0 and step % save_steps == 0:
            model.save_pretrained(run_dir / f"checkpoint-{step}")

    model.save_pretrained(run_dir / "final")
    tokenizer.save_pretrained(run_dir / "final")

    total_hours = (time.monotonic() - start_time) / 3600
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
    print(f"[sft] done. {total_steps} steps in {total_hours:.2f} GPU-hours. Artifacts -> {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if "run_name" not in cfg:
        cfg["run_name"] = Path(args.config).stem
    train(cfg)


if __name__ == "__main__":
    main()
