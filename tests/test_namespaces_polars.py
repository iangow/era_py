import polars as pl
import pytest

import era_pl


def test_winsorize_matches_r_type_2_on_discontinuities():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.winsorize(0.25).alias("x"))

    assert out["x"].to_list() == [1.5, 2.0, 3.0, 3.5]


def test_winsorize_matches_r_type_2_between_discontinuities():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.winsorize(0.30).alias("x"))

    assert out["x"].to_list() == [2.0, 2.0, 3.0, 3.0]


def test_winsorize_respects_grouped_counts_for_type_2():
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 100.0, 1.0, 2.0, 100.0],
        }
    )

    out = df.with_columns(pl.col("x").era.winsorize(1 / 3).over("g").alias("w"))

    assert out["w"].to_list() == [1.5, 2.0, 51.0, 1.5, 2.0, 51.0]


def test_winsorize_matches_r_type_2_with_ties():
    df = pl.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 4.0, 100.0]})

    out_025 = df.select(pl.col("x").era.winsorize(0.25).alias("x"))
    out_0333 = df.select(pl.col("x").era.winsorize(1 / 3).alias("x"))

    assert out_025["x"].to_list() == [2.0, 2.0, 2.0, 3.0, 4.0, 4.0]
    assert out_0333["x"].to_list() == [2.0, 2.0, 2.0, 3.0, 3.5, 3.5]


def test_winsorize_allows_one_sided_tail_override():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.winsorize(prob=0.25, p_high=0.50).alias("x"))

    assert out["x"].to_list() == [1.5, 2.0, 2.5, 2.5]


def test_winsorize_uses_explicit_tails_when_both_are_supplied():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.winsorize(prob=0.01, p_low=0.25, p_high=0.75).alias("x"))

    assert out["x"].to_list() == [1.5, 2.0, 3.0, 3.5]


def test_winsorize_supports_one_sided_when_prob_is_none():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.winsorize(prob=None, p_high=0.50).alias("x"))

    assert out["x"].to_list() == [1.0, 2.0, 2.5, 2.5]


def test_winsorize_prob_is_equivalent_to_explicit_symmetric_tails():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    out_prob = df.select(pl.col("x").era.winsorize(prob=0.01).alias("x"))
    out_explicit = df.select(
        pl.col("x").era.winsorize(p_low=0.01, p_high=0.99).alias("x")
    )

    assert out_prob["x"].to_list() == out_explicit["x"].to_list()


def test_truncate_supports_one_sided_tail_override():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.truncate(prob=0.25, p_high=0.50).alias("x"))

    assert out["x"].to_list() == [None, 2.0, None, None]


def test_truncate_supports_one_sided_when_prob_is_none():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    out = df.select(pl.col("x").era.truncate(prob=None, p_low=0.25).alias("x"))

    assert out["x"].to_list() == [None, 2.0, 3.0, 4.0]


def test_truncate_prob_is_equivalent_to_explicit_symmetric_tails():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    out_prob = df.select(pl.col("x").era.truncate(prob=0.01).alias("x"))
    out_explicit = df.select(
        pl.col("x").era.truncate(p_low=0.01, p_high=0.99).alias("x")
    )

    assert out_prob["x"].to_list() == out_explicit["x"].to_list()


def test_winsorize_requires_at_least_one_cutoff():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="At least one"):
        df.select(pl.col("x").era.winsorize(prob=None).alias("x"))
