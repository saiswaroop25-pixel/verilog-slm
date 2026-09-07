# Why this project checks more than "the testbench passed"

The base implementation guide's harness has two stages: `compile`
(iverilog) and `simulate` (vvp + testbench). That is enough to measure
*functional* correctness, which is what VerilogEval/RTLLM pass@k
measures, and it's the right scope for reproducing that benchmark
faithfully.

It is not enough to claim the model's output is *industry-standard RTL*.
A design can compile cleanly and pass every testbench vector while still
being something no synthesis engineer would accept:

- an `always @(*)` block missing a branch, which simulates fine (the
  simulator keeps the old value, same as a real latch would) but
  synthesizes to a **latch** — usually a bug, always a red flag in review
- two `always` blocks driving the same `reg`, which some simulators
  tolerate via evaluation-order luck but which is a **multi-driven net**
  and not synthesizable / not portable across toolchains
- feedback loops with no registered break — a **combinational loop** that
  simulators may or may not converge on but that real hardware cannot
  implement
- code that leans on simulation-only constructs (`initial` blocks driving
  outputs, delays used for logic timing, etc.) that has no synthesis
  meaning at all

None of that shows up in a `vvp` pass. So this project's harness
(`src/verify/harness.py`) adds two stages between compile and simulate:

1. **lint** (`verible-verilog-lint`, optional/soft-gated) — style and
   naming-convention compliance, the kind of thing that blocks a PR at a
   company with a house Verilog style guide.
2. **synthesize** (`yosys` generic synthesis + `check`, optional/soft-
   gated) — the actual test for latches, multi-driven nets, combinational
   loops, and constructs yosys refuses to synthesize.

Both stages degrade gracefully: if the binary isn't on `PATH` (e.g. a
bare Colab CPU runtime with only `iverilog` installed), the stage is
recorded as `"skipped"` rather than silently treated as a pass, and
`require_lint_clean` / `require_synthesizable` in `verify()` control
whether a dirty result there fails the whole verification or is only
logged. Default: lint is advisory, synthesis-clean is required — a design
that doesn't survive `yosys check` fails verification the same way a
compile error would, which is also what feeds the repair loop.

## What this adds concretely

- **Taxonomy**: three new label families
  (`inferred_latch`, `combinational_loop`, `multi_driven_net`,
  `unsynthesizable_construct`; `lint_naming_convention`,
  `lint_implicit_width`) alongside the guide's compile/semantic labels
  (`src/verify/taxonomy.py`).
- **Repair loop**: synthesis and lint failures get raw tool output fed
  back exactly like a compile error does (`src/infer/repair.py`) — same
  mechanism, one more failure class it can fix.
- **Prompting**: the system prompt (`src/utils/prompts.py`) states the
  house style up front — nonblocking in clocked blocks, blocking in
  combinational blocks, explicit `default` in every `case`, no inferred
  latches, one-line module-purpose comment — so the model is asked for
  industry-clean code from the first sample, not just corrected into it
  after a repair round.
- **Post-processing**: the final accepted (or final attempted) design is
  run through `verible-verilog-format` before being handed back
  (`src/infer/postprocess.py`), so the artifact a reviewer sees matches a
  standard style guide regardless of what whitespace the model sampled.
- **Metrics**: `catch_all_share` and the taxonomy table both fold in the
  new stages, so "how often is the model's output synthesizable" is a
  first-class number next to pass@1, not an afterthought.

## Reportable framing

If you want one sentence for the writeup: *"the field measures
functional correctness against a testbench; we additionally measure
synthesizability and style compliance, because a hardware engineer's bar
for 'this is usable RTL' is higher than 'the simulator agreed with the
reference'."* Report the synthesis-clean rate and lint-clean rate
alongside pass@1 in the same table — it's a second axis nobody in the
cited baselines reports, the same way the guide's per-GPU-hour metric is
a second axis nobody reports.
