import shutil

import pytest

from src.verify.harness import _parse_testbench_result, verify, verify_static

HAS_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def test_parse_testbench_mismatch_count_zero_is_pass():
    assert _parse_testbench_result("Mismatches: 0\n") is True


def test_parse_testbench_mismatch_count_nonzero_is_fail():
    assert _parse_testbench_result("Mismatches: 3\n") is False


def test_parse_testbench_sentinel_pass():
    assert _parse_testbench_result("...\nALL TESTS PASSED\n") is True


def test_parse_testbench_exit_zero_with_failures_is_not_a_pass():
    # the guide's explicit warning: exit code 0 with printed mismatches
    # must not be read as a pass just because $finish fired cleanly.
    assert _parse_testbench_result("running...\nFAIL: output mismatch at time 40\n$finish called\n") is False


def test_parse_testbench_silent_output_is_fail_not_pass():
    assert _parse_testbench_result("$finish called at time 100\n") is False


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog/vvp not installed")
def test_verify_passes_trivial_design():
    code = "module top(output y); assign y = 1'b1; endmodule"
    tb = """
    module tb;
      wire y;
      top dut(.y(y));
      initial begin
        #1;
        if (y !== 1'b1) $display("Mismatches: 1");
        else $display("Mismatches: 0");
        $finish;
      end
    endmodule
    """
    result = verify(code, tb, top_module="top", require_synthesizable=False)
    assert result.ok is True
    assert result.stage == "pass"


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog/vvp not installed")
def test_verify_reports_compile_error():
    code = "module top(output y); assign y = ; endmodule"  # syntax error
    result = verify(code, "module tb; endmodule", top_module="top")
    assert result.ok is False
    assert result.stage == "compile"


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog/vvp not installed")
def test_verify_static_does_not_require_testbench():
    code = "module top(input a, output y); assign y = a; endmodule"
    result = verify_static(code, top_module="top", require_synthesizable=False)
    assert result.stage in ("pass", "compile", "lint", "synthesize")
