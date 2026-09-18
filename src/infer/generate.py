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
from pathlib import Path
from typing import Any

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.prompts import build_prompt
from src.utils.seeding import set_seed


def load_model_for_inference(adapter_path: str, base_model_name: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.bfloat16,
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
    flat_outputs: list[str] = []
    for i in range(0, len(expanded), batch_size):
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

    for idx, text in enumerate(flat_outputs):
        all_outputs[idx // n].append(text)
    return all_outputs


def run(adapter: str, split_path: str, split: str, n: int, temperature: float,
        top_p: float, out_path: str, seed: int = 1337) -> None:
    set_seed(seed)
    rows = [r for r in read_jsonl(split_path) if r["split"] == split]
    model, tokenizer = load_model_for_inference(adapter)

    prompts = [build_prompt(r["instruction"]) for r in rows]
    completions = generate_batch(model, tokenizer, prompts, n, temperature, top_p)

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
    args = ap.parse_args()

    split_path = args.split_path
    if split_path is None:
        import yaml
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            split_path = yaml.safe_load(f)["data"]["corpus_path"]

    run(args.adapter, split_path, args.split, args.n, args.temperature, args.top_p, args.out, args.seed)


if __name__ == "__main__":
    main()
