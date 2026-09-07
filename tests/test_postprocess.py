from src.infer.postprocess import extract_module, finalize, strip_markdown_fences


def test_strip_markdown_fences():
    text = "```verilog\nmodule m; endmodule\n```"
    assert strip_markdown_fences(text) == "module m; endmodule"


def test_strip_markdown_fences_noop_without_fence():
    text = "module m; endmodule"
    assert strip_markdown_fences(text) == text


def test_extract_module_drops_surrounding_prose():
    text = "Here is the module:\nmodule m(input a, output y);\nassign y = a;\nendmodule\nHope this helps!"
    result = extract_module(text)
    assert result.startswith("module m")
    assert result.endswith("endmodule")
    assert "Hope this helps" not in result


def test_finalize_falls_back_to_light_tidy_without_verible():
    raw = "```verilog\nmodule m(input a,\toutput y);\nassign y = a;\nendmodule\n```"
    out = finalize(raw)
    assert "\t" not in out
    assert out.strip().startswith("module m")
    assert out.strip().endswith("endmodule")
