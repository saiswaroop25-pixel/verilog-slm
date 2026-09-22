"""Batched sampling from a trained adapter (Part 6 diagnostic pass, and
the generation side of Part 10 eval).

    python -m src.infer.generate \\
        --adapter artifacts/m0/final --split probe --n 5 \\
        --temperature 0.8 --top_p 0.95 --out artifacts/m0_probe_gens.jsonl

n=5 non-greedy samples per problem for the diagnostic pass (you want the
error *distribution*, greedy gives one point from it); n=20 for the
pass@k eval (Part 10).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.prompts import build_prompt
from src.utils.seeding import set_seed


def load_model_for_inference(adapter_path: str, base_model_name: str | None = None):
    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_dir = Path(adapter_path)
    if base_model_name is None:
        cfg_path = adapter_dir / "adapter_config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_model_name = json.load(f)["base_model_name_or_path"]

    # From the base model, not adapter_dir: LoRA fine-tuning never touches
    # the tokenizer, so a saved-adapter copy is never anything other than
    # a duplicate of the base's -- and only ever a liability (a corrupted
    # or truncated tokenizer.json in a saved checkpoint, seen in practice,
    # otherwise silently blocks inference on an adapter with perfectly
    # good weights).
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Batched generate() on a decoder-only model advances every row in the
    # batch from the same (padded) sequence length -- with right-padding,
    # a shorter prompt's "next token" position lands inside its own
    # padding instead of right after its real content, silently
    # corrupting that row's output. Must be left-padded for generation.
    tokenizer.padding_side = "left"

    # Load the base model exactly the way sft.py trained against it --
    # same configs/base.yaml quantization.* block, the single source of
    # truth for both train and inference. This used to load full-precision
    # bf16 instead ("cleaner inference"), which was wrong: a LoRA delta
    # learned during training implicitly compensates for the specific
    # numerical error 4-bit NF4 quantization introduces into the base
    # weights it was trained against. Apply that same delta to a
    # full-precision, differently-typed base instead and the mismatch
    # doesn't just lose some quality -- confirmed in practice, it produces
    # completely degenerate output (one token repeated for the entire
    # generation), on an adapter whose weights were otherwise verified
    # totally normal (finite, small magnitude).
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    quant = base_cfg.get("quantization", {})
    bnb_config = None
    compute_dtype = torch.bfloat16
    if quant.get("load_in_4bit"):
        compute_dtype = getattr(torch, quant.get("bnb_4bit_compute_dtype", "float16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        )

    # Single GPU, not device_map="auto"'s multi-GPU pipeline split: that
    # split cost training measurably (bs=2 across 2 T4s ran slower than
    # bs=1 on one, see configs/m0_memcheck_bs2.yaml), and generation is
    # far more exposed to it -- with KV-caching, producing each new token
    # is a full forward pass that crosses the pipeline boundary once, so
    # up to max_new_tokens crossings per sequence instead of training's
    # one per batch.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map=device, torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_batch(
    model, tokenizer, prompts: list[str], n: int, temperature: float, top_p: float,
    max_new_tokens: int = 768,
) -> list[list[str]]:
    """Returns, for each prompt, a list of n sampled completions."""
    import torch

    all_outputs: list[list[str]] = [[] for _ in prompts]
    expanded = [p for p in prompts for _ in range(n)]

    batch_size = 8
    n_batches = (len(expanded) + batch_size - 1) // batch_size
    flat_outputs: list[str] = []
    start_time = time.monotonic()
    for batch_idx, i in enumerate(range(0, len(expanded), batch_size)):
        chunk = expanded[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5), top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        flat_outputs.extend(decoded)

        # No output at all until the very end otherwise -- this loop can
        # run for hours (each batch does up to max_new_tokens sequential
        # forward passes), and silence is indistinguishable from a hang.
        elapsed = time.monotonic() - start_time
        done = batch_idx + 1
        avg = elapsed / done
        eta_min = avg * (n_batches - done) / 60
        print(f"[generate] batch {done}/{n_batches} ({elapsed/60:.1f}m elapsed, "
              f"~{eta_min:.1f}m remaining)")

    for idx, text in enumerate(flat_outputs):
        all_outputs[idx // n].append(text)
    return all_outputs


def run(adapter: str, split_path: str, split: str, n: int, temperature: float,
        top_p: float, out_path: str, seed: int = 1337, limit: int | None = None,
        max_new_tokens: int = 768) -> None:
    set_seed(seed)
    rows = [r for r in read_jsonl(split_path) if r["split"] == split]
    if limit is not None and limit < len(rows):
        # A random subsample, not just the first `limit` rows -- corpus
        # rows aren't shuffled by source/tier, so taking a prefix would
        # bias the diagnostic table toward whichever dataset/tier happens
        # to sort first instead of representing the full split.
        import random
        rows = random.Random(seed).sample(rows, limit)
    model, tokenizer = load_model_for_inference(adapter)

    prompts = [build_prompt(r["instruction"]) for r in rows]
    completions = generate_batch(model, tokenizer, prompts, n, temperature, top_p, max_new_tokens)

    out_rows: list[dict[str, Any]] = []
    for row, comps in zip(rows, completions):
        out_rows.append({
            "id": row["id"], "instruction": row["instruction"],
            "tags": row["tags"], "samples": comps,
        })
    write_jsonl(out_path, out_rows)
    print(f"[generate] {len(out_rows)} problems x {n} samples -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split-path", default=None, help="defaults to configs/base.yaml data.corpus_path")
    ap.add_argument("--split", default="probe", choices=["train", "probe"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of problems sampled from the split (random subsample, "
             "not a prefix) -- useful when the full split is too large to run in full",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="batched generate() only finishes a batch once every row in it stops, "
             "so a handful of long/non-terminating completions can dominate wall-clock "
             "time -- lower this for a faster (if more truncation-prone) pass",
    )
    args = ap.parse_args()

    split_path = args.split_path
    if split_path is None:
        import yaml
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            split_path = yaml.safe_load(f)["data"]["corpus_path"]

    run(args.adapter, split_path, args.split, args.n, args.temperature, args.top_p,
        args.out, args.seed, args.limit, args.max_new_tokens)


if __name__ == "__main__":
    main()
