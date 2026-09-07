"""Prompt templates. Kept in one place so the SFT formatting function and
the inference-time prompt builder can never drift apart -- a classic bug
class where train-time and test-time prompts silently diverge.

Also encodes the industry-standard output contract the model is trained
to follow (see docs/industry_standards.md for the full rationale):
  - a one-line module-purpose comment above `module`
  - explicit port directions and widths, one per line
  - nonblocking assignments in every clocked block, blocking in every
    combinational block
  - an explicit `default` branch in every `case`
  - no bare numeric literals for bus widths -- use the declared parameter
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a senior RTL design engineer. Write synthesizable Verilog-2001 "
    "that would pass review at a hardware company: clean `iverilog -Wall` "
    "compile, no inferred latches, no multi-driven nets, explicit `default` "
    "case branches, nonblocking assignments in clocked blocks, blocking "
    "assignments in combinational blocks, and one short comment above the "
    "module stating its purpose. Output only the Verilog module -- no prose, "
    "no markdown fences."
)

GENERATE_TEMPLATE = """{system}

<specification>
{spec}
</specification>

Write the complete Verilog module."""


REPAIR_TEMPLATE = """{system}

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

Rewrite the complete module, fixing this error. Output only Verilog code."""


def build_prompt(spec: str) -> str:
    return GENERATE_TEMPLATE.format(system=SYSTEM_PROMPT, spec=spec)


def build_repair_prompt(spec: str, previous_attempt: str, raw_error_text: str, stage: str) -> str:
    return REPAIR_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        spec=spec,
        previous_attempt=previous_attempt,
        raw_error_text=raw_error_text,
        stage=stage,
    )


def format_sft_example(instruction: str, code: str) -> str:
    """Train-time formatting. Must mirror build_prompt()'s framing so the
    model sees the same instruction shape at train and inference time."""
    return GENERATE_TEMPLATE.format(system=SYSTEM_PROMPT, spec=instruction) + "\n" + code
