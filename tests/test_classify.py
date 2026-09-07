from src.verify.classify import (
    classify_compile_error,
    classify_semantic_failure,
    classify_synth_error,
)
from src.verify.taxonomy import ErrorLabel


def test_classify_syntax_error():
    assert classify_compile_error("design.v:3: syntax error") == ErrorLabel.SYNTAX_ERROR


def test_classify_undefined_module():
    assert classify_compile_error("Unknown module type: foo") == ErrorLabel.UNDEFINED_MODULE


def test_classify_width_mismatch():
    assert classify_compile_error("width mismatch in port connection") == ErrorLabel.WIDTH_MISMATCH


def test_classify_compile_other_fallback():
    assert classify_compile_error("some brand new never-seen-before message") == ErrorLabel.COMPILE_OTHER


def test_classify_none_when_empty():
    assert classify_compile_error("") == ErrorLabel.NONE


def test_classify_synth_latch():
    assert classify_synth_error("Warning: Latch inferred for signal y") == ErrorLabel.INFERRED_LATCH


def test_classify_synth_multi_driven():
    assert classify_synth_error("ERROR: Net 'y' driven by more than one source") == ErrorLabel.MULTI_DRIVEN_NET


def test_semantic_blocking_in_sequential():
    code = """
    module m(input clk, input d, output reg q);
      always @(posedge clk) q = d;
    endmodule
    """
    assert classify_semantic_failure(code, "mismatch") == ErrorLabel.BLOCKING_IN_SEQUENTIAL


def test_semantic_missing_default_case():
    code = """
    module m(input [1:0] sel, output reg y);
      always @(*) begin
        case (sel)
          2'b00: y = 0;
          2'b01: y = 1;
        endcase
      end
    endmodule
    """
    assert classify_semantic_failure(code, "mismatch") == ErrorLabel.MISSING_DEFAULT_CASE


def test_semantic_wrong_logic_fallback():
    code = "module m(input a, output y); assign y = ~a; endmodule"
    assert classify_semantic_failure(code, "mismatch") == ErrorLabel.WRONG_LOGIC_OTHER
