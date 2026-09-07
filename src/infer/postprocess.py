"""Post-generation cleanup applied to whatever the model finally returns
(after the repair loop, if any) before it is handed to a human reviewer.

This is deliberately separate from the verification harness: the harness
decides pass/fail and drives repair, this module never changes semantics
and never runs on the repair-loop hot path. It only runs once, on the
final accepted (or final attempted) design, and it does two things a
professional would expect from a tool claiming "industry-standard output":

1. Strip markdown fences / stray prose the model sometimes wraps code in,
   so the artifact handed back is a clean .v file, not a chat reply.
2. Run `verible-verilog-format` if it's on PATH, so indentation/spacing
   matches Google's open-source Verilog style guide rather than whatever
   the model happened to sample. Falls back to a light regex tidy-up
   (trailing whitespace, tabs->spaces) if the binary isn't installed --
   never blocks on a missing tool, since we don't want a formatting nice-
   to-have to break the pipeline on a bare Colab runtime.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_FENCE_RE = re.compile(r"^```(?:verilog|v)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.MULTILINE)


def strip_markdown_fences(text: str) -> str:
    match = _FENCE_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_module(text: str) -> str:
    """If the model emitted anything before/after the module (a common
    instruction-tuned-model habit: "Here is the module:\n```..."), keep
    only from the first `module` to the matching final `endmodule`."""
    start = text.find("module ")
    if start == -1:
        start = text.find("module\n")
    end = text.rfind("endmodule")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start:end + len("endmodule")].strip()


def _light_tidy(code: str) -> str:
    lines = [line.rstrip().replace("\t", "    ") for line in code.splitlines()]
    # collapse 3+ blank lines to 1
    tidied: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        tidied.append(line)
    return "\n".join(tidied).strip() + "\n"


def format_with_verible(code: str) -> str:
    if not shutil.which("verible-verilog-format"):
        return _light_tidy(code)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "design.v"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["verible-verilog-format", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
        except subprocess.TimeoutExpired:
            pass
    return _light_tidy(code)


def finalize(raw_model_output: str) -> str:
    """Full post-processing pipeline: fence-strip -> module-extract ->
    format. This is what infer/repair.py calls on the returned attempt
    before writing it to disk or handing it back to the caller."""
    code = strip_markdown_fences(raw_model_output)
    code = extract_module(code)
    return format_with_verible(code)
