# Verilog SLM — Complete Implementation Guide

Self-mined curriculum + compiler-feedback repair for resource-efficient Verilog generation.

---

## Part 0 — Connecting Claude Code to Google Colab

There is **no official Claude Code ↔ Colab integration**. Claude Code runs on your machine; Colab's GPU is a remote VM. You bridge them. Three options, in order of how much I'd recommend them.

### Option A — Repo-driven (recommended)

This is the most reliable and what I'd use for a graded project. Claude Code never touches the GPU; it writes code, you run it.

```
Your laptop                     GitHub                    Colab
┌──────────────┐              ┌────────┐            ┌──────────────┐
│ Claude Code  │ ── push ──▶  │  repo  │ ── pull ─▶ │ notebook +   │
│ writes code  │              │        │            │ GPU runtime  │
│ reads logs   │ ◀── push ──  │        │ ◀─ push ── │ writes logs  │
└──────────────┘              └────────┘            └──────────────┘
```

Setup:

```bash
# On your laptop
mkdir verilog-slm && cd verilog-slm
git init
gh repo create verilog-slm --private --source=. --remote=origin
claude          # start Claude Code here
```

In Colab, one bootstrap cell at the top of every notebook:

```python
import os, subprocess
from google.colab import drive
drive.mount('/content/drive')

REPO = "https://<TOKEN>@github.com/<you>/verilog-slm.git"
if not os.path.exists('/content/verilog-slm'):
    !git clone {REPO} /content/verilog-slm
%cd /content/verilog-slm
!git pull

# persistent artifacts live on Drive, NOT in the repo
CKPT = '/content/drive/MyDrive/verilog-slm/checkpoints'
LOGS = '/content/drive/MyDrive/verilog-slm/logs'
os.makedirs(CKPT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
```

Loop: Claude Code edits → you `git push` → Colab `git pull` → run → logs land on Drive → you paste the failing traceback back into Claude Code. This is fast in practice because the edit-debug cycles are short and the training runs are long.

### Option B — SSH tunnel (Claude Code drives the GPU directly)

Colab has no inbound SSH, so you tunnel out with Cloudflare. In a Colab cell:

```python
!pip -q install colab-ssh
!curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="<pick-one>")
```

That prints a hostname. Put it in `~/.ssh/config` on your laptop, then Claude Code can run remote commands:

```
Host colab
  HostName <printed>.trycloudflare.com
  User root
  ProxyCommand cloudflared access ssh --hostname %h
```

```bash
ssh colab "cd /content/verilog-slm && python -m src.train.sft --config configs/baseline.yaml"
```

Honest caveats: the tunnel dies when the runtime recycles (~12h max, often sooner), the hostname changes every restart, and long-running commands over SSH will drop. Use `tmux` on the Colab side if you go this route. Good for interactive debugging, bad for launching 6-hour training runs.

### Option C — Colab local runtime (inverse direction)

Run Jupyter on your own machine and connect Colab's frontend to it. Only useful if you have a local GPU — which defeats the point here. Skip.

### What I'd actually do

Option A as the backbone, Option B temporarily when you're debugging something GPU-specific and the paste-the-traceback loop gets tedious. Never launch training over the SSH tunnel — launch it from a notebook cell with `nohup` and checkpointing to Drive.

**Non-negotiable Colab hygiene:**
- Checkpoint adapters to Drive every `save_steps` (200–500). Sessions die without warning.
- Log every run as append-only JSONL on Drive: `{run_id, step, loss, lr, wall_clock, gpu}`.
- Record `nvidia-smi -L` at the start of every run — you need the GPU model for the per-GPU-hour metric to mean anything.
- Pin every dependency in `requirements.txt`. Colab silently upgrades packages between sessions and `transformers` breaks LoRA loading regularly.

---

## Part 1 — Repository structure

```
verilog-slm/
├── configs/
│   ├── base.yaml                 # shared: model, tokenizer, seq len, seed
│   ├── m0_baseline.yaml          # flat SFT
│   └── m1_curriculum.yaml        # curriculum SFT (same budget as m0)
├── src/
│   ├── data/
│   │   ├── build_corpus.py       # load, normalise, dedup, tag, split
│   │   ├── tagger.py             # structural + construct tagging
│   │   └── splits.py             # train / probe / (eval is external)
│   ├── verify/
│   │   ├── harness.py            # compile + simulate, returns VerifyResult
│   │   ├── classify.py           # error string -> taxonomy label
│   │   └── taxonomy.py           # the label set (frozen enum)
│   ├── train/
│   │   ├── sft.py                # single entry point, config-driven
│   │   └── curriculum.py         # builds the weighted/ordered sampler
│   ├── infer/
│   │   ├── generate.py           # batched sampling
│   │   └── repair.py             # the bounded online loop
│   └── eval/
│       ├── run_eval.py           # 2x2 driver
│       └── metrics.py            # pass@k, error rates, efficiency
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_data.ipynb
│   ├── 02_train.ipynb
│   ├── 03_diagnose.ipynb
│   └── 04_eval.ipynb
├── artifacts/                    # gitignored; mirrors to Drive
└── requirements.txt
```

Everything is config-driven and seeded. The single most important property of this codebase: **M0 and M1 must differ in exactly one thing — the sampler.** Same base checkpoint, same optimizer, same LR schedule, same total optimizer steps, same seed, same sequence length. If anything else differs, your ablation is uninterpretable.

---

## Part 2 — Model selection

### Recommended ladder

| Model | Params | 4-bit VRAM (train) | Colab tier | Role in project |
|---|---|---|---|---|
| **Qwen2.5-Coder-1.5B-Instruct** | 1.5B | ~6 GB | T4 | Primary workhorse. Fast iteration, full 2×2 fits easily. |
| **Qwen2.5-Coder-3B-Instruct** | 3B | ~9 GB | T4 (tight) / L4 | Headline result. Best quality-per-hour in this range. |
| **DeepSeek-Coder-6.7B-Instruct** | 6.7B | ~16 GB | L4 / A100 | Stretch goal — **directly comparable to RTLCoder**, which fine-tunes this exact base. |
| **Qwen2.5-Coder-0.5B** | 0.5B | ~3 GB | T4 | Debug only. Pipeline smoke tests in minutes, not hours. |

Alternatives if Qwen is unavailable: `StarCoder2-3B` (Apache 2.0, used by CraftRTL's 15B sibling so the family has Verilog precedent), `CodeGemma-2B`. Both are fine; Qwen2.5-Coder generally has the stronger code prior at these sizes.

### Why the DeepSeek-6.7B option matters

RTLCoder is a fine-tune of DeepSeek-Coder-6.7B. If you run your pipeline on the same base, you get an apples-to-apples statement: *"same base model, same benchmark, ours uses curriculum + repair, theirs uses flat SFT + quality scoring."* That is a far stronger claim than comparing a 1.5B model to a published 6.7B number. If you have any A100 budget at all, spend it here.

### Practical plan

Develop and debug on 0.5B. Run the full 2×2 on 1.5B. Run the headline 2×2 on 3B. If A100 hours allow, run one 6.7B pair for the direct RTLCoder comparison.

### QLoRA configuration

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16     # float16 on T4 (no bf16 support)
  bnb_4bit_use_double_quant: true

lora:
  r: 32                                 # 16 for 0.5B, 32 for 1.5-3B, 64 for 6.7B
  alpha: 64                             # keep alpha = 2r
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  seq_len: 2048
  per_device_batch_size: 4              # 2 on T4 for 3B
  grad_accum_steps: 8                   # keep effective batch = 32 constant
  lr: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.03
  epochs: 3
  gradient_checkpointing: true
  bf16: true                            # fp16 on T4
  seed: 1337
```

Target all seven projection modules, not just attention. On small models the MLP adapters matter more than they do at 15B — attention-only LoRA underfits noticeably below 3B.

Effective batch size is held at 32 across every configuration. When you change hardware and have to change `per_device_batch_size`, change `grad_accum_steps` inversely. Otherwise M0-on-T4 vs M1-on-L4 is not a controlled comparison.

---

## Part 3 — Data pipeline

### Sources

| Dataset | Size | Use |
|---|---|---|
| RTLCoder | 27,000+ instruction–Verilog pairs | Train |
| MG-Verilog | 11,000+ modules × 3 description levels | Train |
| VerilogEval v2 (spec-to-RTL) | 156 problems | **Eval only** |
| RTLLM v2 | 50 designs | **Eval only** |

### Build steps

**1. Normalise.** Everything becomes one schema:

```python
{
  "id": "rtlcoder_00412",
  "source": "rtlcoder",
  "instruction": "Design a 4-bit synchronous counter with async reset...",
  "code": "module counter(...); ... endmodule",
  "tags": {...},          # filled by tagger
  "split": "train"        # train | probe
}
```

**2. Deduplicate — do this properly.** Both corpora are GPT-assisted and contain near-duplicates. Exact hash dedup is not enough.

```python
# MinHash-LSH over normalised code (strip comments, collapse whitespace,
# canonicalise identifiers) at Jaccard threshold 0.85
from datasketch import MinHash, MinHashLSH
```

Then run the same near-dup check **against the eval sets** and drop any training example that collides. VerilogEval is derived from HDLBits, and HDLBits-style problems absolutely appear in GitHub-scraped corpora. If you skip this step, your pass@1 is inflated and a sharp panelist will ask about it. Log how many you dropped — it's a real finding worth a line in your report.

**3. Tag every example.** Two independent tag families.

*Structural tier* (drives ordering):

| Tier | Definition | Detection |
|---|---|---|
| T1 combinational | no `always @(posedge` | AST/regex |
| T2 sequential | has clocked always, no case-FSM | regex + heuristic |
| T3 FSM | `case` inside clocked always with state reg | pattern |
| T4 multi-module | ≥2 `module` decls or instantiation | count |

*Construct tags* (drive reweighting) — multi-label, one example can carry several:

`blocking_assign`, `nonblocking_assign`, `case_stmt`, `param_width`, `generate_block`, `sensitivity_list`, `async_reset`, `sync_reset`, `bit_slicing`, `arithmetic`, `memory_array`

Use a real parser where you can. `pyverilog` gives you an AST; fall back to regex only for tags the AST can't express. Regex-only tagging is noisy enough to blur the very effect you're trying to measure.

**4. Split.** `train` 90% / `probe` 10%, stratified by structural tier so the probe set isn't accidentally all combinational. The probe split is what M0 gets diagnosed on. It is drawn from the training distribution, never from eval.

---

## Part 4 — The verification harness

This is the single most reused component. Everything else calls it.

```python
@dataclass
class VerifyResult:
    stage: str            # "compile" | "simulate" | "pass"
    ok: bool
    error_text: str       # raw iverilog/vvp stderr, unmodified
    error_label: str      # taxonomy label, or "none"
    wall_ms: int
```

### Compile stage

```bash
iverilog -g2012 -Wall -o /tmp/out.vvp design.v testbench.v
```

Run every generation in a fresh temp dir, with a hard timeout (10s compile). Capture stderr verbatim — you need the raw text for the repair loop, and the classifier operates on it separately.

### Simulate stage

```bash
timeout 30 vvp /tmp/out.vvp
```

Two hard requirements people get wrong here:

- **Timeout is mandatory.** A generated FSM with a missing transition will hang forever in simulation. Without a timeout your eval job dies silently at 3am.
- **Parse the testbench's own pass/fail protocol.** VerilogEval and RTLLM testbenches print structured mismatch counts. Do not infer pass/fail from exit code alone — a testbench can exit 0 having printed 400 mismatches.

Run simulations in a process pool. On a Colab CPU you get 2–8 cores; simulation is CPU-bound and embarrassingly parallel, so this is a 4–8× speedup on eval wall-clock for free.

### Error classification

```python
COMPILE_RULES = [
    (r"syntax error",                          "syntax_error"),
    (r"Unknown module type",                   "undefined_module"),
    (r"Unable to bind wire/reg/memory",        "undeclared_identifier"),
    (r"port .* is not a port of",              "port_mismatch"),
    (r"[Ww]idth mismatch|operand .* width",    "width_mismatch"),
    (r"Malformed statement",                   "malformed_statement"),
    (r"has already been declared",             "duplicate_declaration"),
]
```

Semantic labels (compiles but fails simulation) can't come from a regex on stderr — they come from static inspection of the generated code plus the mismatch pattern:

`blocking_in_sequential`, `incomplete_sensitivity`, `missing_default_case`, `reset_polarity`, `off_by_one_width`, `wrong_logic_other`

`wrong_logic_other` is your catch-all. **Report its share.** If 70% of your semantic failures land in the catch-all, your taxonomy isn't discriminating and the curriculum has little to work with — that's an honest, reportable finding, not a failure.

### Validate the classifier before you trust it

Hand-label 100 random failures. Compute classifier agreement. If it's below ~85%, fix the rules before you build a curriculum on top of it. Put this number in your report — it's exactly the kind of methodological detail that separates a careful project from a hand-wavy one.

---

## Part 5 — Baseline M0

Nothing clever here, and that's the point.

```bash
python -m src.train.sft --config configs/m0_baseline.yaml
```

- Shuffled order, seed fixed
- Record: total optimizer steps `S`, wall-clock, GPU model, peak VRAM
- Save adapter + the exact resolved config to Drive

`S` is now a **hard constraint** on M1. Write it into `m1_curriculum.yaml` explicitly rather than deriving it from epochs — with a reweighted sampler, "3 epochs" no longer means the same number of steps.

---

## Part 6 — Diagnostic pass (the offline loop, closed)

```bash
python -m src.infer.generate \
  --adapter artifacts/m0 --split probe --n 5 --temperature 0.8 --top_p 0.95
python -m src.verify.run --input artifacts/m0_probe_gens.jsonl
```

Five samples per probe problem, sampled not greedy — you want the model's *error distribution*, and greedy decoding shows you one point from it.

Output is the taxonomy table:

| label | count | share of failures | tier concentration |
|---|---|---|---|
| syntax_error | 412 | 21% | T1 38%, T3 41% |
| incomplete_sensitivity | 388 | 20% | T2 71% |
| missing_default_case | 297 | 15% | T3 84% |
| width_mismatch | 251 | 13% | T1 55% |
| ... | | | |

The "tier concentration" column is what makes the curriculum non-trivial. A label concentrated in one tier tells you *where* to intervene, not just *what* the model gets wrong.

**This table is a deliverable in its own right.** Nobody has published a per-category failure profile for small Verilog models. Put it in your report whether or not the curriculum works.

---

## Part 7 — Curriculum builder

Two mechanisms, composed.

**Ordering (structural).** Split training into three phases across the fixed step budget `S`:

| Phase | Steps | Tier mix |
|---|---|---|
| 1 | 0.3 S | T1 70%, T2 30% |
| 2 | 0.4 S | T1 25%, T2 40%, T3 35% |
| 3 | 0.3 S | T1 15%, T2 25%, T3 35%, T4 25% |

**Reweighting (error-driven).** Per-example sampling weight:

```python
w(x) = 1.0 + alpha * sum(
    failure_share[t] for t in x.construct_tags
)
# alpha tuned so max weight lands ~3.0; clip to [1.0, 3.0]
```

An example tagged `case_stmt` + `sensitivity_list`, when both are high-failure categories, gets sampled ~3× as often. An example whose constructs the model already handles stays at 1.0.

Implement as a `WeightedRandomSampler` with a per-phase index pool. **The step count does not change** — you are changing *which* examples fill those steps, not how many steps there are. This is the property that keeps the ablation clean.

Log the realised category histogram of what M1 actually saw versus what M0 saw. You'll want that plot; it's the direct evidence that the curriculum did what you designed it to do.

---

## Part 8 — Curriculum M1

```bash
python -m src.train.sft --config configs/m1_curriculum.yaml   # max_steps = S, seed = 1337
```

Assert at startup that `max_steps == S` and that the base checkpoint hash matches M0's. Fail loudly if not. This one assertion prevents the most common way this kind of experiment gets quietly invalidated.

---

## Part 9 — The inference repair loop (online loop)

```python
def generate_with_repair(model, spec, K=3, tb=None):
    trace = []
    code = model.generate(build_prompt(spec))

    for attempt in range(K + 1):
        result = verify(code, tb)
        trace.append({"attempt": attempt, "stage": result.stage,
                      "label": result.error_label})
        if result.ok:
            return code, True, trace
        if attempt == K:
            break
        code = model.generate(
            build_repair_prompt(spec, code, result.error_text, result.stage)
        )
    return code, False, trace
```

Repair prompt template:

```
The following Verilog module was written for this specification:

<spec>
{spec}
</spec>

<code>
{previous_attempt}
</code>

It failed at the {stage} stage with this error:

<error>
{raw_error_text}
</error>

Rewrite the complete module, fixing this error. Output only Verilog code.
```

Design points that make this a *valid* loop rather than blind resampling:

- The error text is **raw and unmodified**. Line numbers and identifier names are the whole signal. Summarising it throws away what makes the loop work.
- Only the **previous** attempt is included, not the full history. Accumulating failed attempts fills context and, empirically, biases the model toward repeating them.
- `K` is bounded. Log the attempt index at which each success occurred — the distribution over `{1st try, 2nd, 3rd, 4th, never}` is a result, not just plumbing.
- **Compile errors and simulation errors take different branches.** A compile error gets the compiler stderr. A simulation mismatch gets the mismatch summary, not the compiler output. Same loop, different feedback payload.
- **The loop is verification-gated, not model-judged.** The model never decides whether it succeeded; `iverilog`/`vvp` does. This is what makes the feedback signal trustworthy — there's no self-assessment failure mode.

Also record **repair-induced regressions**: cases where attempt *n* compiled but attempt *n+1* did not. If that number is non-trivial, "return the last attempt" is the wrong policy and you should return the best-scoring attempt instead. Worth checking, and worth a sentence in the report either way.

---

## Part 10 — Evaluation

### The 2×2

| | Repair off | Repair on |
|---|---|---|
| **M0 baseline** | A — control | B — isolates repair |
| **M1 curriculum** | C — isolates curriculum | D — full pipeline |

Effects you can then state precisely:
- Curriculum effect: `C − A`
- Repair effect: `B − A`
- Interaction: `(D − C) − (B − A)` — is repair *still* worth it after curriculum training, or does the curriculum already prevent the errors repair would have caught?

That interaction term is the most interesting number in the project and nobody has published it.

### Metrics, in full

**1. pass@k — unbiased estimator, not naive**

Generate `n=20` samples per problem, count `c` correct, then:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

```python
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

Report **pass@1** and **pass@5**. Naive pass@k (generate exactly k, check any) has high variance and is not comparable to published numbers. Every paper you cite uses the unbiased form.

Note on the repair-on condition: report it as *pass rate at a fixed budget*, and state the budget explicitly ("1 sample, up to 3 repairs"). Comparing "pass@1 with repair" to "pass@1 without repair" is not a like-for-like comparison unless you say what the compute budget was on each side — repair costs extra forward passes. Be upfront; this is a strength, not a weakness, because you also report the efficiency metric below.

**2. Syntax-error rate**

`compile_failures / total_generations`. Report separately for first attempt and post-repair. This is the metric most sensitive to the curriculum and it moves before pass@1 does — useful early signal during development.

**3. Per-category error rate — your differentiator**

For every taxonomy label, error rate under M0 vs M1:

| label | M0 | M1 | Δ | targeted? |
|---|---|---|---|---|
| incomplete_sensitivity | 20.1% | 12.4% | −7.7 | yes (3× weight) |
| missing_default_case | 15.3% | 9.8% | −5.5 | yes (3× weight) |
| width_mismatch | 13.0% | 12.6% | −0.4 | no (1× weight) |

**This table is the core scientific claim.** If the targeted categories improve and the untargeted ones don't, the mechanism worked *as designed* — that's a causal story, not just a number going up. If everything improves uniformly, the curriculum acted as generic regularisation rather than targeting, which is a different (and honest) finding.

**4. Repair-loop diagnostics**

- Distribution over resolution attempt: `{1, 2, 3, 4, unresolved}`
- Repair success rate by initial error label — which errors are actually fixable from a compiler message?
- Mean extra forward passes per problem (the cost side of the ledger)
- Regression rate (compiled → broke)

**5. Efficiency — the metric the field doesn't report**

```
accuracy per GPU-hour = pass@1 / training_GPU_hours
```

Report the full breakdown, not a single number:

| Model | Train GPU-h | GPU | pass@1 | pass@1 / GPU-h |
|---|---|---|---|---|
| M0 1.5B | 4.2 | T4 | — | — |
| M1 1.5B | 4.2 | T4 | — | — |
| M1 3B | 9.8 | L4 | — | — |
| RTLCoder-6.7B (published) | not reported | — | 34% | not computable |

Two caveats you must state explicitly, because a good panelist will raise them:
- GPU-hours across different GPU models aren't directly comparable. Normalise by reporting GPU-model alongside, or convert to a rough FLOP estimate.
- Published baselines don't report their training cost, so the comparison is one-sided. Say so. The point isn't "we beat them per hour" — it's "this axis is unmeasured in the literature and here is a first measurement."

**6. Statistical honesty**

- Run each configuration with **3 seeds**. Report mean ± std.
- With 156 + 50 = 206 eval problems, a 2-point pass@1 difference is well inside noise. Compute a bootstrap 95% CI over problems.
- If the curriculum effect is smaller than the seed variance, say that plainly. CraftRTL's own gains were 3.8–10.9% at 15B; at 1.5B with a different mechanism, a 2–4% effect is a plausible and reportable outcome — but only if you can distinguish it from noise.

---

## Part 11 — Compute budget

Rough estimates for Colab Pro (100 compute units/month).

| Task | Hardware | Hours |
|---|---|---|
| Pipeline debug (0.5B) | T4 | 2 |
| M0 + M1, 1.5B, 3 seeds | T4 | ~26 |
| Diagnostic pass (1.5B) | T4 | 1.5 |
| M0 + M1, 3B, 1 seed | L4 | ~20 |
| Eval, all 4 cells × 20 samples × 3 seeds | T4/L4 | ~12 |
| Buffer | — | 15 |
| **Total** | | **~77 h** |

The 6.7B DeepSeek pair adds ~15 A100-hours and burns units far faster — treat it as a stretch goal contingent on remaining budget, not a dependency.

Simulation is CPU-bound. Run verification passes on a **CPU-only** Colab runtime to avoid spending GPU units on `iverilog`. This alone saves meaningful budget.

---

## Part 12 — Timeline mapped to your slides

| Slide milestone | Date | Work |
|---|---|---|
| Design | 24 Aug | Repo scaffold, harness, tagger, classifier validated on 100 hand-labels |
| Implementation | 3 Oct | Data pipeline, M0 + M1 trained at 1.5B, diagnostic table produced |
| Testing | 25 Oct | Full 2×2 × 3 seeds, 3B run, all metrics computed |
| Document | 30 Nov | Analysis, per-category table, efficiency comparison, writeup |

---

## Part 13 — Failure modes and what to do

| Risk | Signal | Response |
|---|---|---|
| Curriculum shows no gain | `C − A` inside seed noise | Report as negative result **with** the per-category table showing weights didn't shift errors. This is publishable analysis, not a dead end. |
| Taxonomy dominated by catch-all | `wrong_logic_other` > 60% | Refine semantic rules, or narrow claim to compile-stage errors only and say so. |
| Contamination found | Near-dup hits vs eval | Drop them, report the count. Finding this is a contribution. |
| Colab session loss | Run dies mid-training | Already mitigated by Drive checkpointing every 200 steps. |
| 3B doesn't fit on T4 | OOM | Drop `per_device_batch_size` to 2, raise `grad_accum` to 16, keep effective batch at 32. |
| Repair loop regressions | Attempt n+1 worse than n | Switch return policy from "last" to "best verified". |

---

## The one-line summary of the whole design

Two loops, both closed by a real verifier rather than by the model's own judgement: an **offline loop** where M0's measured failures reweight the corpus that trains M1 under an identical step budget, and an **online loop** where compiler and simulator output drives bounded repair at inference. The 2×2 ablation separates their individual and joint contributions, and every result is reported both as raw accuracy and per GPU-hour.
