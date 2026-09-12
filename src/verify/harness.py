"""The verification harness. Everything else in the project calls this.

Pipeline: compile (iverilog) -> lint (verible, optional) ->
synthesize (yosys, optional) -> simulate (vvp).

The lint and synthesize stages are this project's addition on top of a
plain compile+simulate harness. Rationale: a testbench only tells you the
design is *functionally* correct for the vectors it happens to check. It
says nothing about whether the RTL would survive a real synthesis flow
(latches, multi-driven nets, combinational loops) or whether it would pass
review at a company with a house style guide (naming, explicit widths).
"Industry-standard RTL" means clean on all four stages, not just "the
testbench prints PASS."

Both extra stages are soft-gated by config: `require_lint_clean` and
`require_synthesizable` default to False so the harness still runs on a
machine that only has iverilog installed (e.g. a bare Colab CPU runtime).
When the binaries aren't found, the stage is skipped and recorded as
"skipped" rather than silently treated as a pass.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.verify.classify import (
    classify_compile_error,
    classify_lint_findings,
    classify_semantic_failure,
    classify_synth_error,
)
from src.verify.taxonomy import ErrorLabel, Stage

COMPILE_TIMEOUT_S = 10
SIMULATE_TIMEOUT_S = 30
SYNTH_TIMEOUT_S = 20
LINT_TIMEOUT_S = 10


@dataclass
class VerifyResult:
    stage: str            # "compile" | "lint" | "synthesize" | "simulate" | "pass"
    ok: bool
    error_text: str       # raw tool stderr/stdout, unmodified
    error_label: str      # taxonomy label, or "none"
    wall_ms: int
    stage_results: dict = field(default_factory=dict)  # per-stage detail, for logging


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str, bool]:
    """Returns (returncode, stdout, stderr, timed_out)."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", (e.stderr or "") + "\n[TIMEOUT]", True


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def _parse_testbench_result(stdout: str) -> bool:
    """Parse the VerilogEval / RTLLM-style pass/fail protocol.

    These testbenches print explicit mismatch counts and a final verdict
    string; a zero exit code alone is NOT sufficient (a testbench can exit
    0 having printed hundreds of mismatches). We look for the conventions
    used across VerilogEval v2 / RTLLM v2 harnesses -- verified against the
    live testbench sources of both benchmarks, not guessed:
      - VerilogEval v2 spec-to-rtl: "Mismatches: N in M samples" -> ok iff N==0
      - RTLLM v2: "===========Your Design Passed===========" (dash count and
        spacing vary per-problem -- checked with flexible whitespace, no
        anchoring to a specific run of '=') on success; failure messages are
        NOT standardized across RTLLM's 50 testbenches (seen: "Test
        completed with N/100 failures", "Test failed: ...", "Failed at...",
        bare "Error: ..." lines), so failure is the absence of the pass
        sentinel, not a positive failure match.
      - a "$finish" firing at all is NOT taken as pass; absence of any
        recognised sentinel is treated as a failure so silent designs
        don't get credited.
    """
    mismatch_match = re.search(r"[Mm]ismatch(?:es)?(?:\s*count)?\s*[:=]\s*(\d+)", stdout)
    if mismatch_match:
        return int(mismatch_match.group(1)) == 0

    if re.search(r"ALL\s*TESTS?\s*PASSED|TESTBENCH\s*PASSED|Hint:\s*Total\s*mismatched\s*samples\s*is\s*0", stdout, re.IGNORECASE):
        return True

    if re.search(r"your\s+design\s+passed", stdout, re.IGNORECASE):
        return True

    return False


def compile_stage(design_path: Path, tb_path: Path, work_dir: Path) -> tuple[bool, str, int]:
    if not _tool_available("iverilog"):
        raise RuntimeError(
            "iverilog not found on PATH. Unlike lint/synthesize, compile is not "
            "soft-gated -- it's the one tool this harness cannot run without. "
            "Install it (`apt-get install iverilog` on Colab, see notebooks/00_setup.ipynb)."
        )
    out_vvp = work_dir / "out.vvp"
    start = time.monotonic()
    rc, stdout, stderr, timed_out = _run(
        ["iverilog", "-g2012", "-Wall", "-o", str(out_vvp), str(design_path), str(tb_path)],
        cwd=work_dir, timeout=COMPILE_TIMEOUT_S,
    )
    wall_ms = int((time.monotonic() - start) * 1000)
    ok = (rc == 0) and not timed_out and out_vvp.exists()
    text = stderr if not ok else (stdout + stderr)
    return ok, text, wall_ms


def lint_stage(design_path: Path, work_dir: Path) -> tuple[bool, str, int, str]:
    """Returns (clean, findings_text, wall_ms, status) where status is
    'ran' | 'skipped' (tool missing)."""
    if not _tool_available("verible-verilog-lint"):
        return True, "", 0, "skipped"
    start = time.monotonic()
    rc, stdout, stderr, timed_out = _run(
        ["verible-verilog-lint", str(design_path)],
        cwd=work_dir, timeout=LINT_TIMEOUT_S,
    )
    wall_ms = int((time.monotonic() - start) * 1000)
    findings = stdout + stderr
    clean = (rc == 0) and not timed_out and not findings.strip()
    return clean, findings, wall_ms, "ran"


_SYNTH_SCRIPT = """
read_verilog -sv {design}
hierarchy -check -top {top}
proc
opt
memory -nomap
techmap
opt
check
"""


def synthesize_stage(design_path: Path, top_module: str, work_dir: Path) -> tuple[bool, str, int, str]:
    """Generic-cell synthesis + `check` via yosys. This is not a PPA-grade
    ASIC/FPGA flow; it's a cheap, tool-based sanity pass that catches the
    classes of bug a testbench routinely misses: inferred latches,
    multi-driven nets, combinational feedback loops, and constructs yosys
    simply refuses to synthesize."""
    if not _tool_available("yosys"):
        return True, "", 0, "skipped"
    script_path = work_dir / "synth.ys"
    script_path.write_text(
        _SYNTH_SCRIPT.format(design=design_path.name, top=top_module), encoding="utf-8"
    )
    start = time.monotonic()
    rc, stdout, stderr, timed_out = _run(
        ["yosys", "-Q", "-T", "-s", str(script_path)],
        cwd=work_dir, timeout=SYNTH_TIMEOUT_S,
    )
    wall_ms = int((time.monotonic() - start) * 1000)
    text = stdout + stderr
    ok = (rc == 0) and not timed_out and not re.search(
        r"Latch inferred|found a latch|combinational loop|multiple conflicting drivers|ERROR",
        text,
    )
    return ok, text, wall_ms, "ran"


def simulate_stage(work_dir: Path) -> tuple[bool, str, int, bool]:
    out_vvp = work_dir / "out.vvp"
    start = time.monotonic()
    rc, stdout, stderr, timed_out = _run(
        ["vvp", str(out_vvp)], cwd=work_dir, timeout=SIMULATE_TIMEOUT_S,
    )
    wall_ms = int((time.monotonic() - start) * 1000)
    if timed_out:
        return False, stdout + stderr, wall_ms, True
    ok = _parse_testbench_result(stdout)
    return ok, stdout + stderr, wall_ms, False


def verify_static(
    code: str,
    top_module: str = "top",
    require_lint_clean: bool = False,
    require_synthesizable: bool = True,
    work_root: str | None = None,
) -> VerifyResult:
    """Compile + lint + synthesize only, no simulate stage.

    Exists because the M0 diagnostic pass (Part 6) runs over the *probe*
    split, which is drawn from the training corpora (RTLCoder / MG-Verilog).
    Those corpora ship instruction->code pairs, not instruction->(code,
    testbench) triples, so there is no oracle to simulate against for an
    arbitrary training example -- only VerilogEval/RTLLM (eval-only, per
    Part 3) ship testbenches. Silently treating a missing testbench as a
    simulate-stage failure would mislabel every generation and pollute
    the taxonomy table with a spurious spike on whatever semantic label
    the classifier defaults to.

    Using this function instead means the M0 diagnostic table only
    reports compile/lint/synthesis-stage taxonomy on the probe split --
    an honest limitation, not a gap papered over. Functional (simulate-
    stage / semantic-label) failure rates are measured where testbenches
    actually exist: the eval sets, via `verify()` inside run_eval.py.
    """
    total_start = time.monotonic()
    stage_results: dict = {}

    with tempfile.TemporaryDirectory(dir=work_root, prefix="verify_") as tmp:
        work_dir = Path(tmp)
        design_path = work_dir / "design.v"
        design_path.write_text(code, encoding="utf-8")

        # iverilog needs something to elaborate against; an empty stub
        # top-level instantiation is enough to catch syntax/elaboration
        # errors in the design itself without requiring a real testbench.
        stub_tb = work_dir / "testbench.v"
        stub_tb.write_text(
            f"module _stub_tb;\ninitial begin\n  $finish;\nend\nendmodule\n",
            encoding="utf-8",
        )

        ok, text, wall_ms = compile_stage(design_path, stub_tb, work_dir)
        stage_results["compile"] = {"ok": ok, "wall_ms": wall_ms, "status": "ran"}
        if not ok:
            label = classify_compile_error(text)
            return VerifyResult(Stage.COMPILE.value, False, text, label.value,
                                 int((time.monotonic() - total_start) * 1000), stage_results)

        clean, findings, wall_ms, status = lint_stage(design_path, work_dir)
        stage_results["lint"] = {"ok": clean, "wall_ms": wall_ms, "status": status}
        if status == "ran" and not clean and require_lint_clean:
            label = classify_lint_findings(findings)
            return VerifyResult(Stage.LINT.value, False, findings, label.value,
                                 int((time.monotonic() - total_start) * 1000), stage_results)

        synth_ok, synth_text, wall_ms, status = synthesize_stage(design_path, top_module, work_dir)
        stage_results["synthesize"] = {"ok": synth_ok, "wall_ms": wall_ms, "status": status}
        total_ms = int((time.monotonic() - total_start) * 1000)
        if status == "ran" and not synth_ok and require_synthesizable:
            label = classify_synth_error(synth_text)
            return VerifyResult(Stage.SYNTHESIZE.value, False, synth_text, label.value, total_ms, stage_results)

        return VerifyResult(Stage.PASS.value, True, "", ErrorLabel.NONE.value, total_ms, stage_results)


def verify(
    code: str,
    testbench: str,
    top_module: str = "top",
    require_lint_clean: bool = False,
    require_synthesizable: bool = True,
    work_root: str | None = None,
) -> VerifyResult:
    """Run one generated design through the full stage pipeline.

    Every call gets a fresh temp dir (never reuse across generations —
    stale .vvp / synth artifacts from a previous design are a classic
    source of false passes).
    """
    total_start = time.monotonic()
    stage_results: dict = {}

    with tempfile.TemporaryDirectory(dir=work_root, prefix="verify_") as tmp:
        work_dir = Path(tmp)
        design_path = work_dir / "design.v"
        tb_path = work_dir / "testbench.v"
        design_path.write_text(code, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")

        ok, text, wall_ms = compile_stage(design_path, tb_path, work_dir)
        stage_results["compile"] = {"ok": ok, "wall_ms": wall_ms, "status": "ran"}
        if not ok:
            label = classify_compile_error(text)
            return VerifyResult(Stage.COMPILE.value, False, text, label.value,
                                 int((time.monotonic() - total_start) * 1000), stage_results)

        clean, findings, wall_ms, status = lint_stage(design_path, work_dir)
        stage_results["lint"] = {"ok": clean, "wall_ms": wall_ms, "status": status}
        if status == "ran" and not clean and require_lint_clean:
            label = classify_lint_findings(findings)
            return VerifyResult(Stage.LINT.value, False, findings, label.value,
                                 int((time.monotonic() - total_start) * 1000), stage_results)

        synth_ok, synth_text, wall_ms, status = synthesize_stage(design_path, top_module, work_dir)
        stage_results["synthesize"] = {"ok": synth_ok, "wall_ms": wall_ms, "status": status}
        if status == "ran" and not synth_ok and require_synthesizable:
            label = classify_synth_error(synth_text)
            return VerifyResult(Stage.SYNTHESIZE.value, False, synth_text, label.value,
                                 int((time.monotonic() - total_start) * 1000), stage_results)

        ok, sim_text, wall_ms, timed_out = simulate_stage(work_dir)
        stage_results["simulate"] = {"ok": ok, "wall_ms": wall_ms, "status": "ran", "timed_out": timed_out}
        total_ms = int((time.monotonic() - total_start) * 1000)
        if timed_out:
            return VerifyResult(Stage.SIMULATE.value, False, sim_text, ErrorLabel.TIMEOUT.value,
                                 total_ms, stage_results)
        if not ok:
            label = classify_semantic_failure(code, sim_text)
            return VerifyResult(Stage.SIMULATE.value, False, sim_text, label.value,
                                 total_ms, stage_results)

        return VerifyResult(Stage.PASS.value, True, "", ErrorLabel.NONE.value, total_ms, stage_results)
