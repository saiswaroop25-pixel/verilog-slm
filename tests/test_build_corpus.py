from src.data.build_corpus import _dedup_exact_hash, canonicalise_for_hashing, normalize_schema


def test_normalize_schema_maps_common_field_names():
    row = normalize_schema({"prompt": "do X", "response": "module m; endmodule"}, "src", 0)
    assert row["id"] == "src_00000"
    assert row["instruction"] == "do X"
    assert row["code"] == "module m; endmodule"
    assert row["split"] == "train"


def test_canonicalise_strips_comments_and_renames_identifiers():
    a = "// header\nmodule foo(input clk, output reg q); always @(posedge clk) q <= 1; endmodule"
    b = "module bar(input xyz, output reg qq); always @(posedge xyz) qq <= 1; endmodule"
    ca, cb = canonicalise_for_hashing(a), canonicalise_for_hashing(b)
    assert ca == cb  # same structure modulo identifier naming -> should canonicalise identically


def test_canonicalise_preserves_keywords():
    code = "module m(input a, output b); assign b = a; endmodule"
    canon = canonicalise_for_hashing(code)
    assert "module" in canon and "endmodule" in canon
    assert "assign" in canon


def test_dedup_exact_hash_drops_identical_code():
    rows = [
        {"id": "a", "code": "module m; endmodule"},
        {"id": "b", "code": "module m; endmodule"},
        {"id": "c", "code": "module n; wire w; endmodule"},
    ]
    kept, dropped = _dedup_exact_hash(rows)
    assert dropped == 1
    assert len(kept) == 2
