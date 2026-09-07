from src.data.tagger import tag_example
from src.verify.taxonomy import StructuralTier


def test_combinational_tier():
    code = "module m(input a, input b, output y); assign y = a & b; endmodule"
    tags = tag_example(code, prefer_ast=False)
    assert tags.tier == StructuralTier.T1_COMBINATIONAL.value


def test_sequential_tier():
    code = """
    module m(input clk, input d, output reg q);
      always @(posedge clk) q <= d;
    endmodule
    """
    tags = tag_example(code, prefer_ast=False)
    assert tags.tier == StructuralTier.T2_SEQUENTIAL.value
    assert "nonblocking_assign" in tags.constructs


def test_fsm_tier():
    code = """
    module m(input clk, input rst, output reg [1:0] state);
      always @(posedge clk) begin
        case (state)
          2'b00: state <= 2'b01;
          default: state <= 2'b00;
        endcase
      end
    endmodule
    """
    tags = tag_example(code, prefer_ast=False)
    assert tags.tier == StructuralTier.T3_FSM.value
    assert "case_stmt" in tags.constructs


def test_multi_module_tier():
    code = """
    module sub(input a, output y); assign y = a; endmodule
    module top(input a, output y);
      sub s0(.a(a), .y(y));
    endmodule
    """
    tags = tag_example(code, prefer_ast=False)
    assert tags.tier == StructuralTier.T4_MULTI_MODULE.value


def test_construct_tags_bit_slicing_and_arithmetic():
    code = "module m(input [7:0] a, output [7:0] y); assign y = a[3:0] + 1; endmodule"
    tags = tag_example(code, prefer_ast=False)
    assert "bit_slicing" in tags.constructs
    assert "arithmetic" in tags.constructs
