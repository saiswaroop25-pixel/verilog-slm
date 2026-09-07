# Verilog SLM

Self-mined curriculum + compiler-feedback repair for resource-efficient,
**industry-standard** Verilog generation from small language models.

Two loops, both closed by a real verifier rather than by the model's own
judgement:

- an **offline loop**: M0 (flat-SFT baseline)'s measured failures reweight
  the training corpus that trains M1 (curriculum) under an identical step
  budget
- an **online loop**: compiler, linter, and synthesizer output drives a
  bounded repair loop at inference time

A 2x2 ablation (baseline/curriculum x repair-off/repair-on) separates
their individual and joint contributions. Every result is reported both
as raw accuracy and per GPU-hour.

See `Verilog SLM - Implementation Guide.md` for the full design rationale
this repo implements, and `docs/industry_standards.md` for what this
implementation adds on top of that guide and why.

## What "industry-standard output" means here

A testbench pass only proves *functional* correctness against the vectors
it happens to check. It says nothing about whether the RTL would survive
a real synthesis flow or a code review. So the verification harness here
runs four stages, not two:

```
compile (iverilog) -> lint (verible, optional) -> synthesize (yosys, optional) -> simulate (vvp)
```

The lint/synthesize stages catch inferred latches, multi-driven nets,
combinational loops, and style violations that a testbench alone would
miss. They're soft-gated so the repo still works with only `iverilog`
installed. The model is also prompted with an explicit house-style
contract (`src/utils/prompts.py`) and every final output is run through
`verible-verilog-format` before being handed back (`src/infer/postprocess.py`).
Full rationale: `docs/industry_standards.md`.

## Repository layout

```
configs/          model/training config, layered (base -> m0 / m1)
src/data/         corpus build: normalise, MinHash dedup, tag, split
src/verify/       the 4-stage harness + error taxonomy + classifier
src/train/        QLoRA SFT entry point + curriculum sampler
src/infer/        batched generation + bounded repair loop + post-processing
src/eval/         2x2 driver, diagnostic-table builder, metrics
scripts/          classifier validation, eval-set adapters
notebooks/        Colab-side driver notebooks, 00 through 04
tests/            pure-Python unit tests (no GPU/iverilog required)
```

## Quickstart (local, CPU-only side of the pipeline)

```bash
pip install -r requirements.txt
pytest -q          # 48 tests, no external tools required
```

Building the corpus, training, and evaluation all need real RTL/data
tooling and a GPU (`requirements-train.txt`) — see Part 0 of the
implementation guide for the Colab bridging setup, and `notebooks/00_setup.ipynb`
onward for the driven workflow:

```bash
python -m src.data.build_corpus --sources rtlcoder=data/raw/rtlcoder.jsonl mg-verilog=data/raw/mg_verilog.jsonl \
    --eval-sets data/eval/verilogeval_v2.jsonl data/eval/rtllm_v2.jsonl --out artifacts/corpus.jsonl

python -m src.train.sft --config configs/m0_baseline.yaml
python -m src.infer.generate --adapter artifacts/m0/final --split probe --n 5 --out artifacts/m0_probe_gens.jsonl
python -m src.eval.diagnose --gens artifacts/m0_probe_gens.jsonl --out artifacts/m0_diagnostic.json
python -m src.train.sft --config configs/m1_curriculum.yaml   # asserts against m0/run_meta.json

python -m src.eval.run_eval --m0-adapter artifacts/m0/final --m1-adapter artifacts/m1/final \
    --eval-sets data/eval/verilogeval_v2.jsonl data/eval/rtllm_v2.jsonl \
    --n 20 --k-repair 3 --seeds 1337 2025 7 --out artifacts/eval_report.json
```

## Non-negotiable invariants (enforced in code, not just docs)

- **M0 and M1 differ in exactly one thing: the sampler.** `configs/base.yaml`
  is the single source of truth for model/LoRA/optimizer/seed/seq-len;
  `sft.py`'s `assert_matches_m0` fails loudly if M1's base-checkpoint hash
  or step count diverge from M0's recorded run.
- **Eval sets are never trained on.** `build_corpus.py` MinHash-checks the
  training pool against the eval sets and drops any collision, logging the
  count.
- **The repair loop is verification-gated.** The model never judges its
  own output; `iverilog`/`yosys`/`vvp` do.
- **pass@k is the unbiased estimator**, not naive generate-k-check-any, and
  every headline number ships a bootstrap 95% CI over problems and a
  mean±std across 3 seeds.
