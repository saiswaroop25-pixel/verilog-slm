"""Map raw tool output (or generated code + mismatch text) to a taxonomy label.

Three independent rule sets, one per stage that can fail before simulation,
plus a semantic classifier for simulation-stage mismatches that cannot be
read off stderr and instead comes from static inspection of the code.

Validate these rules against hand labels before trusting them (see
Part 4 of the implementation guide / scripts/validate_classifier.py).
Target agreement: >=85%.
"""

from __future__ import annotations

import re

from src.verify.taxonomy import ErrorLabel

# ---------------------------------------------------------------------------
# Stage 1: compile (iverilog stderr)
# ---------------------------------------------------------------------------
COMPILE_RULES: list[tuple[str, ErrorLabel]] = [
    (r"syntax error", ErrorLabel.SYNTAX_ERROR),
    (r"Unknown module type", ErrorLabel.UNDEFINED_MODULE),
    (r"Unable to bind wire/reg/memory", ErrorLabel.UNDECLARED_IDENTIFIER),
    (r"port .* is not a port of", ErrorLabel.PORT_MISMATCH),
    (r"[Ww]idth mismatch|operand .* width", ErrorLabel.WIDTH_MISMATCH),
    (r"Malformed statement", ErrorLabel.MALFORMED_STATEMENT),
    (r"has already been declared", ErrorLabel.DUPLICATE_DECLARATION),
]

# ---------------------------------------------------------------------------
# Stage 2: lint (verible-verilog-lint output, one finding per line, format:
#   file:line:col: message [rule-name]
# ---------------------------------------------------------------------------
LINT_RULES: list[tuple[str, ErrorLabel]] = [
    (r"\[module-filename\]|\[signal-name-style\]|\[parameter-name-style\]|\[module-name-style\]",
     ErrorLabel.LINT_NAMING),
    (r"\[explicit-parameter-storage-type\]|\[net-variable-bit-width\]",
     ErrorLabel.LINT_EXPLICIT_WIDTH),
]

# ---------------------------------------------------------------------------
# Stage 3: synthesize (yosys stdout/stderr from `synth` + `check`)
# ---------------------------------------------------------------------------
SYNTH_RULES: list[tuple[str, ErrorLabel]] = [
    (r"Latch inferred|found a latch", ErrorLabel.INFERRED_LATCH),
    (r"combinational loop|Found logic loop", ErrorLabel.COMBINATIONAL_LOOP),
    (r"multiple conflicting drivers|multi-driven|Net .* driven by more than one",
     ErrorLabel.MULTI_DRIVEN_NET),
    (r"can't synthesize|Unsupported|not supported in synthesis",
     ErrorLabel.UNSYNTHESIZABLE_CONSTRUCT),
]


def _first_match(text: str, rules: list[tuple[str, ErrorLabel]], other: ErrorLabel) -> ErrorLabel:
    for pattern, label in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return other


def classify_compile_error(stderr_text: str) -> ErrorLabel:
    if not stderr_text.strip():
        return ErrorLabel.NONE
    return _first_match(stderr_text, COMPILE_RULES, ErrorLabel.COMPILE_OTHER)


def classify_lint_findings(lint_text: str) -> ErrorLabel:
    if not lint_text.strip():
        return ErrorLabel.NONE
    return _first_match(lint_text, LINT_RULES, ErrorLabel.LINT_STYLE_OTHER)


def classify_synth_error(synth_text: str) -> ErrorLabel:
    if not synth_text.strip():
        return ErrorLabel.NONE
    return _first_match(synth_text, SYNTH_RULES, ErrorLabel.SYNTH_OTHER)


# ---------------------------------------------------------------------------
# Stage 4: semantic (simulate stage) — regex on stderr is not enough here,
# these come from static inspection of the *generated code* combined with
# whatever mismatch text the testbench printed.
# ---------------------------------------------------------------------------

_ALWAYS_STAR_BLOCK = re.compile(
    r"always\s*@\s*\(\s*\*\s*\)\s*(?:begin)?(.*?)(?:end\s*(?:always)?|\Z)",
    re.DOTALL,
)
_ALWAYS_EXPLICIT_SENS = re.compile(
    r"always\s*@\s*\(([^)]*)\)\s*(?:begin)?(.*?end)",
    re.DOTALL,
)
_CLOCKED_ALWAYS = re.compile(
    r"always\s*@\s*\(\s*(pos|neg)edge\s+\w+.*?\)\s*(?:begin)?(.*?)(?:end\s*(?:always)?)",
    re.DOTALL,
)
_CASE_BLOCK = re.compile(r"\bcase\s*\(.*?\)(.*?)endcase", re.DOTALL)
_ASYNC_RESET = re.compile(r"always\s*@\s*\(\s*(pos|neg)edge\s+\w+\s*,?\s*(?:or\s+)?(pos|neg)edge\s+(\w*rst\w*|\w*reset\w*)", re.IGNORECASE)


def _rhs_identifiers(expr: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", expr))


def _combinational_incomplete_sensitivity(code: str) -> bool:
    """Explicit-list combinational always blocks (no posedge/negedge) whose
    sensitivity list is missing an identifier that's actually read in the
    body -> classic simulation/synthesis mismatch source."""
    for match in _ALWAYS_EXPLICIT_SENS.finditer(code):
        sens_list, body = match.group(1), match.group(2)
        if re.search(r"posedge|negedge", sens_list, re.IGNORECASE):
            continue  # clocked block, not combinational
        sens_ids = _rhs_identifiers(sens_list) - {"or"}
        if not sens_ids:
            continue
        # identifiers read on the RHS of assignments / conditions in body
        rhs_text = re.sub(r"^\s*\w+\s*(<=|=)", "", body, flags=re.MULTILINE)
        body_ids = _rhs_identifiers(body)
        keywords = {"begin", "end", "if", "else", "case", "endcase", "default"}
        missing = (body_ids - sens_ids - keywords)
        # crude but effective: an identifier assigned to (LHS) shouldn't count
        lhs_ids = set(re.findall(r"(\w+)\s*(?:<=|=)", body))
        missing -= lhs_ids
        if missing:
            return True
    return False


def _case_missing_default(code: str) -> bool:
    for match in _CASE_BLOCK.finditer(code):
        body = match.group(1)
        if "default" not in body:
            return True
    return False


def _blocking_in_sequential(code: str) -> bool:
    for match in _CLOCKED_ALWAYS.finditer(code):
        body = match.group(2)
        # ignore nested combinational always blocks accidentally captured
        if re.search(r"(?<!<)(?<![<>=!])\b\w+\s*=\s*(?!=)", body):
            return True
    return False


def classify_semantic_failure(code: str, mismatch_text: str) -> ErrorLabel:
    """Best-effort static classification of a compile+synth-clean design
    that failed simulation. Order matters: check the most specific,
    highest-precision signals first and fall back to the catch-all.
    """
    if _blocking_in_sequential(code):
        return ErrorLabel.BLOCKING_IN_SEQUENTIAL
    if _combinational_incomplete_sensitivity(code):
        return ErrorLabel.INCOMPLETE_SENSITIVITY
    if _case_missing_default(code):
        return ErrorLabel.MISSING_DEFAULT_CASE
    if re.search(r"reset|rst", mismatch_text, re.IGNORECASE) and _ASYNC_RESET.search(code):
        return ErrorLabel.RESET_POLARITY
    if re.search(r"width|bit\s*\d+.*expected|truncat", mismatch_text, re.IGNORECASE):
        return ErrorLabel.OFF_BY_ONE_WIDTH
    return ErrorLabel.WRONG_LOGIC_OTHER
