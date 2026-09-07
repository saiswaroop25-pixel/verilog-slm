"""Frozen error-taxonomy label set.

Every failure produced anywhere in the pipeline (compile, lint, synthesis,
simulate) is reduced to exactly one label from this module. The set is
intentionally frozen: adding a label mid-project invalidates comparisons
between runs collected before/after the change, so treat this file as an
append-only contract and never rename or delete an existing label.

Stages, in the order the harness runs them:
  compile    -> iverilog parses/elaborates the design
  lint       -> verible-verilog-lint checks style / industry coding rules
  synthesize -> yosys attempts a generic synthesis pass (catches latches,
                combinational loops, multi-driven nets: things that compile
                and simulate fine but would never survive a real ASIC/FPGA
                flow). This stage is the project's main addition on top of
                the guide's compile+simulate harness, because "simulates
                correctly" and "synthesizable, industry-clean RTL" are not
                the same claim.
  simulate   -> vvp runs the testbench and the harness parses its own
                pass/fail protocol
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    COMPILE = "compile"
    LINT = "lint"
    SYNTHESIZE = "synthesize"
    SIMULATE = "simulate"
    PASS = "pass"


class ErrorLabel(str, Enum):
    # --- compile stage: regex-derived from iverilog stderr -----------------
    SYNTAX_ERROR = "syntax_error"
    UNDEFINED_MODULE = "undefined_module"
    UNDECLARED_IDENTIFIER = "undeclared_identifier"
    PORT_MISMATCH = "port_mismatch"
    WIDTH_MISMATCH = "width_mismatch"
    MALFORMED_STATEMENT = "malformed_statement"
    DUPLICATE_DECLARATION = "duplicate_declaration"
    COMPILE_OTHER = "compile_other"

    # --- lint stage: style / industry coding-standard violations -----------
    # Non-fatal by default (configurable to be gating) — these are what
    # distinguish RTL a professional would accept in review from RTL that
    # merely simulates.
    LINT_NAMING = "lint_naming_convention"
    LINT_EXPLICIT_WIDTH = "lint_implicit_width"
    LINT_STYLE_OTHER = "lint_style_other"

    # --- synthesis stage: yosys-derived -------------------------------------
    INFERRED_LATCH = "inferred_latch"
    COMBINATIONAL_LOOP = "combinational_loop"
    MULTI_DRIVEN_NET = "multi_driven_net"
    UNSYNTHESIZABLE_CONSTRUCT = "unsynthesizable_construct"
    SYNTH_OTHER = "synth_other"

    # --- semantic labels: compiles + synthesizes, fails simulation ---------
    BLOCKING_IN_SEQUENTIAL = "blocking_in_sequential"
    INCOMPLETE_SENSITIVITY = "incomplete_sensitivity"
    MISSING_DEFAULT_CASE = "missing_default_case"
    RESET_POLARITY = "reset_polarity"
    OFF_BY_ONE_WIDTH = "off_by_one_width"
    WRONG_LOGIC_OTHER = "wrong_logic_other"

    # --- bookkeeping ---------------------------------------------------------
    TIMEOUT = "timeout"
    NONE = "none"


COMPILE_LABELS = {
    ErrorLabel.SYNTAX_ERROR,
    ErrorLabel.UNDEFINED_MODULE,
    ErrorLabel.UNDECLARED_IDENTIFIER,
    ErrorLabel.PORT_MISMATCH,
    ErrorLabel.WIDTH_MISMATCH,
    ErrorLabel.MALFORMED_STATEMENT,
    ErrorLabel.DUPLICATE_DECLARATION,
    ErrorLabel.COMPILE_OTHER,
}

LINT_LABELS = {
    ErrorLabel.LINT_NAMING,
    ErrorLabel.LINT_EXPLICIT_WIDTH,
    ErrorLabel.LINT_STYLE_OTHER,
}

SYNTH_LABELS = {
    ErrorLabel.INFERRED_LATCH,
    ErrorLabel.COMBINATIONAL_LOOP,
    ErrorLabel.MULTI_DRIVEN_NET,
    ErrorLabel.UNSYNTHESIZABLE_CONSTRUCT,
    ErrorLabel.SYNTH_OTHER,
}

SEMANTIC_LABELS = {
    ErrorLabel.BLOCKING_IN_SEQUENTIAL,
    ErrorLabel.INCOMPLETE_SENSITIVITY,
    ErrorLabel.MISSING_DEFAULT_CASE,
    ErrorLabel.RESET_POLARITY,
    ErrorLabel.OFF_BY_ONE_WIDTH,
    ErrorLabel.WRONG_LOGIC_OTHER,
}

CATCH_ALL_LABELS = {
    ErrorLabel.COMPILE_OTHER,
    ErrorLabel.LINT_STYLE_OTHER,
    ErrorLabel.SYNTH_OTHER,
    ErrorLabel.WRONG_LOGIC_OTHER,
}


class StructuralTier(str, Enum):
    T1_COMBINATIONAL = "T1"
    T2_SEQUENTIAL = "T2"
    T3_FSM = "T3"
    T4_MULTI_MODULE = "T4"


CONSTRUCT_TAGS = (
    "blocking_assign",
    "nonblocking_assign",
    "case_stmt",
    "param_width",
    "generate_block",
    "sensitivity_list",
    "async_reset",
    "sync_reset",
    "bit_slicing",
    "arithmetic",
    "memory_array",
)
