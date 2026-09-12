"""Structural + construct tagging (Part 3, step 3 of the guide).

Uses pyverilog's AST when it's importable and can parse the snippet
(training corpora contain plenty of malformed/truncated modules that
pyverilog chokes on); falls back to regex for anything the AST can't
express or when parsing fails. Regex-only tagging is noisy enough to
blur the curriculum effect, so prefer the AST path whenever it's available
and log how often we fall back -- that fallback rate is worth reporting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.verify.taxonomy import CONSTRUCT_TAGS, StructuralTier

_CLOCKED_ALWAYS = re.compile(r"always\s*@\s*\(\s*(pos|neg)edge\b", re.IGNORECASE)
_CASE_IN_CLOCKED = re.compile(
    r"always\s*@\s*\([^)]*(pos|neg)edge[^)]*\)(?:begin)?(.*?)(?:end\s*(?:always)?)",
    re.DOTALL | re.IGNORECASE,
)
_MODULE_DECL = re.compile(r"\bmodule\s+\w+", re.IGNORECASE)
# "type-name instance-name (" -- matches both a real instantiation ("sub s0(")
# and, spuriously, a module *header* itself ("module counter (" parses as
# type="module", name="counter"). Filter the latter out by keyword below --
# don't drop the pattern's MULTILINE anchor, that's what lets it match a
# port-list header without matching mid-statement port declarations.
_MODULE_INSTANCE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+(?:#\([^)]*\)\s*)?[A-Za-z_]\w*\s*\(", re.MULTILINE
)
_NOT_INSTANCE_TYPE_KEYWORDS = {
    "module", "input", "output", "inout", "wire", "reg", "assign", "always",
    "if", "else", "case", "endcase", "for", "while", "function", "endfunction",
    "parameter", "localparam", "generate", "endgenerate", "begin", "end",
    "initial", "genvar", "task", "endtask",
}


def _count_module_instances(code: str) -> int:
    return sum(
        1 for m in _MODULE_INSTANCE.finditer(code)
        if m.group(1).lower() not in _NOT_INSTANCE_TYPE_KEYWORDS
    )

_CONSTRUCT_PATTERNS: dict[str, str] = {
    "blocking_assign": r"(?<![<>=!])\b\w+\s*=\s*(?!=)",
    "nonblocking_assign": r"<=",
    "case_stmt": r"\bcase\s*\(",
    "param_width": r"\bparameter\s+.*\[\s*[\w:\-\+ ]+\s*\]",
    "generate_block": r"\bgenerate\b",
    "sensitivity_list": r"always\s*@\s*\(",
    "async_reset": r"always\s*@\s*\([^)]*(pos|neg)edge[^)]*,?\s*(?:or\s+)?(pos|neg)edge[^)]*rst",
    "sync_reset": r"if\s*\(\s*!?\s*\w*rst\w*\s*\)",
    "bit_slicing": r"\w+\s*\[\s*[\w\-\+: ]+\s*:\s*[\w\-\+ ]+\s*\]",
    "arithmetic": r"[^=!<>]\+|\-(?!>)|\*|<<|>>",
    "memory_array": r"\breg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]",
}


@dataclass
class Tags:
    tier: str
    constructs: list[str] = field(default_factory=list)
    tagger_backend: str = "regex"


def _detect_tier(code: str) -> str:
    n_modules = len(_MODULE_DECL.findall(code)) + _count_module_instances(code)
    if n_modules >= 2:
        return StructuralTier.T4_MULTI_MODULE.value

    for match in _CASE_IN_CLOCKED.finditer(code):
        body = match.group(2)
        if re.search(r"\bcase\s*\(", body):
            return StructuralTier.T3_FSM.value

    if _CLOCKED_ALWAYS.search(code):
        return StructuralTier.T2_SEQUENTIAL.value

    return StructuralTier.T1_COMBINATIONAL.value


def _detect_constructs_regex(code: str) -> list[str]:
    found = []
    for tag in CONSTRUCT_TAGS:
        pattern = _CONSTRUCT_PATTERNS.get(tag)
        if pattern and re.search(pattern, code):
            found.append(tag)
    return found


_pyverilog_parser = None  # lazy singleton -- see _get_pyverilog_parser()


def _get_pyverilog_parser():
    """pyverilog's top-level `parse()` convenience function builds a brand
    new VerilogParser (and therefore a brand new PLY yacc() grammar/LALR
    table) on every single call, and its default `debug=True` prints
    "Generating LALR tables" / "WARNING: N shift/reduce conflicts" to
    stderr each time too. Called once per training example (tens of
    thousands of times over a real corpus), that's both a severe,
    easily-missed performance bug -- confirmed by direct measurement: two
    separate VerilogParser() constructions each re-trigger the warning,
    while one shared instance reused across parses triggers it zero times
    -- and a wall of console spam that looks like a hang. Build the parser
    exactly once per process (silently, debug=False) and reuse it for
    every call instead.
    """
    global _pyverilog_parser
    if _pyverilog_parser is None:
        from pyverilog.vparser.parser import VerilogParser
        _pyverilog_parser = VerilogParser(debug=False)
    return _pyverilog_parser


def _try_pyverilog_tags(code: str) -> list[str] | None:
    """Return construct tags derived from a real AST, or None if pyverilog
    isn't installed or fails to parse this snippet."""
    try:
        import subprocess
        import tempfile
        from pathlib import Path
        from pyverilog.vparser.parser import VerilogPreprocessor
    except ImportError:
        return None

    try:
        parser = _get_pyverilog_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.v"
            path.write_text(code, encoding="utf-8")
            # Preprocessing (macro/ifdef expansion) is inherently per-file --
            # unlike the parser/grammar above, this step is cheap (one
            # `iverilog -E` subprocess call) and isn't worth caching.
            #
            # VerilogPreprocessor.preprocess() runs this via bare
            # `subprocess.call(cmd)`, which inherits our stdout/stderr and
            # floods the console with one "warning: macro X undefined" line
            # per undefined-macro reference -- expected and harmless on
            # RTLCoder/MG-Verilog (standalone snippets routinely reference
            # parameters from an original file's now-absent `include`), but
            # alarming at corpus scale. Build the same command it would
            # (reusing its cmd-building logic rather than duplicating the
            # iverilog-path/include/define handling) and run it ourselves
            # with output suppressed; a genuine preprocessing failure still
            # surfaces the same way -- as a non-zero-effect parse that this
            # function's outer `except Exception: return None` falls back
            # to regex tagging for.
            pp_output = str(Path(tmp) / "preprocess.output")
            preprocessor = VerilogPreprocessor([str(path)], pp_output)
            subprocess.run(preprocessor.iv + list(preprocessor.filelist),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for temp_file_path in preprocessor.temp_files_paths:
                Path(temp_file_path).unlink(missing_ok=True)
            text = Path(pp_output).read_text(encoding="utf-8")
        ast = parser.parse(text, debug=0)
    except Exception:
        return None

    found: set[str] = set()
    try:
        from pyverilog.vparser.ast import (
            Case, GenerateStatement, NonblockingSubstitution,
            BlockingSubstitution, Parameter, Partselect,
        )

        def walk(node):
            if isinstance(node, NonblockingSubstitution):
                found.add("nonblocking_assign")
            elif isinstance(node, BlockingSubstitution):
                found.add("blocking_assign")
            elif isinstance(node, Case):
                found.add("case_stmt")
            elif isinstance(node, GenerateStatement):
                found.add("generate_block")
            elif isinstance(node, Partselect):
                found.add("bit_slicing")
            for c in node.children():
                walk(c)

        walk(ast)
    except Exception:
        return None

    # a few tags are cheaper / more reliable via regex even in the AST path
    for tag in ("param_width", "sensitivity_list", "async_reset", "sync_reset",
                "arithmetic", "memory_array"):
        pattern = _CONSTRUCT_PATTERNS.get(tag)
        if pattern and re.search(pattern, code):
            found.add(tag)

    return sorted(found)


def tag_example(code: str, prefer_ast: bool = True) -> Tags:
    tier = _detect_tier(code)

    if prefer_ast:
        ast_tags = _try_pyverilog_tags(code)
        if ast_tags is not None:
            return Tags(tier=tier, constructs=ast_tags, tagger_backend="pyverilog")

    return Tags(tier=tier, constructs=_detect_constructs_regex(code), tagger_backend="regex")
