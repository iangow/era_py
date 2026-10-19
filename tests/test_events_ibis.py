import datetime as dt

import numpy as np
import pandas as pd
import pytest

ibis = pytest.importorskip("ibis")
duckdb = pytest.importorskip("duckdb")

from era_py import events


@pytest.fixture
def event_returns(tmp_path, monkeypatch):
    folder = tmp_path / "crsp"
    folder.mkdir()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    conn = ibis.duckdb.connect(":memory:")

    def run(stock=(0.1, 0.2), market=None, size=None, dlret=None, event_data=None, **kwargs):
        n = len(stock)
        dates = [dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)][:n]
        market = [0.03, 0.04, 0.05][:n] if market is None else market
        size = [0.02, 0.01, 0.06][:n] if size is None else size
        dlret = [None] * n if dlret is None else dlret
        tables = {
            "dsf": {"permno": [10001] * n, "date": dates,
                    "ret": np.asarray(stock, dtype=float)},
            "dsi": {"date": dates, "vwretd": np.asarray(market, dtype=float)},
            "erdport1": {"permno": [10001] * n, "date": dates,
                         "decret": np.asarray(size, dtype=float)},
            "dsedelist": {"permno": [10001] * n, "dlstdt": dates,
                          "dlret": np.asarray(dlret, dtype=float)},
        }
        for name, columns in tables.items():
            pd.DataFrame(columns).to_parquet(folder / f"{name}.parquet", index=False)
        if event_data is None:
            event_data = pd.DataFrame({
                "permno": [10001], "event_date": dates[:1], "end_event_date": dates[-1:],
            })
        return events.get_event_rets(
            event_data, conn, end_event_date="end_event_date", data_dir=tmp_path, **kwargs
        )

    yield run
    conn.disconnect()


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({}, [0.32, 0.2488, 0.2898]),
        ({"stock": [0.1, -0.2]}, [-0.12, -0.1912, -0.1502]),
        ({"stock": [0, 0]}, [0, -0.0712, -0.0302]),
        ({"stock": [0.1, None], "market": [0, 0], "size": [0, 0]}, [0.1] * 3),
        ({"market": [0.03, None], "size": [None, 0.01]}, [0.32, 0.29, 0.31]),
        ({"market": [None, None], "size": [None, None]}, [0.32, np.nan, np.nan]),
        ({"stock": [-1, 0.2]}, [-1, -1.0712, -1.0302]),
        ({"stock": [-1, -1]}, [-1, -1.0712, -1.0302]),
        ({"stock": [-1, None], "market": [0, 0], "size": [0, 0]}, [-1] * 3),
        ({"market": [-1, 0.04]}, [0.32, 1.32, 0.2898]),
        ({"size": [0.02, -1]}, [0.32, 0.2488, 1.32]),
        ({"market": [-1, None], "size": [None, -1]}, [0.32, 1.32, 1.32]),
        ({"dlret": [None, -1]}, [-1, -1.0712, -1.0302]),
        ({"stock": [0.1, None], "dlret": [None, -1]}, [-1, -1.0712, -1.0302]),
        ({"stock": [-1, 0.2], "market": [-1, 0.04], "size": [0.02, -1]}, [-1, 0, 0]),
    ],
    ids=[
        "ordinary", "mixed-sign", "zero", "partial-stock", "partial-benchmarks",
        "missing-benchmarks", "stock-loss", "repeated-loss", "loss-and-missing",
        "market-loss", "size-loss", "benchmark-losses-and-missing", "delisting-loss",
        "delisting-with-missing-stock", "stock-and-benchmark-losses",
    ],
)
def test_event_returns_compounding_and_missingness(event_returns, inputs, expected):
    actual = event_returns(**inputs)

    assert len(actual) == 1
    np.testing.assert_allclose(
        actual[["ret_raw", "ret_mkt", "ret_sz"]].iloc[0].to_numpy(),
        expected, rtol=1e-12, atol=1e-12, equal_nan=True,
    )


@pytest.mark.parametrize("field", ["stock", "market", "size"])
@pytest.mark.parametrize("returns", [[-1.1, 0.2], [-1, -1.1]])
def test_event_returns_reject_below_minus_one(event_returns, field, returns):
    with pytest.raises(duckdb.OutOfRangeException, match="logarithm of a negative number"):
        event_returns(**{field: returns})


def test_event_returns_all_missing_stock_remains_empty(event_returns):
    assert event_returns(stock=[None, None]).empty


def test_event_returns_unmatched_security_remains_empty(event_returns):
    data = pd.DataFrame({
        "permno": [99999], "event_date": [dt.date(2020, 1, 2)],
        "end_event_date": [dt.date(2020, 1, 3)],
    })

    assert event_returns(event_data=data).empty


def test_event_returns_preserve_offsets_and_suffix(event_returns):
    data = pd.DataFrame({
        "permno": [10001], "event_date": [dt.date(2020, 1, 3)],
        "end_event_date": [dt.date(2020, 1, 3)],
    })

    actual = event_returns(
        stock=[0.1, -1, 0.2], event_data=data, win_start=-1, win_end=1, suffix="_window"
    )

    np.testing.assert_allclose(
        actual[["ret_raw_window", "ret_mkt_window", "ret_sz_window"]].iloc[0],
        [-1, -np.prod(1 + np.array([0.03, 0.04, 0.05])),
         -np.prod(1 + np.array([0.02, 0.01, 0.06]))],
    )
    assert actual["event_date"][0] == pd.Timestamp("2020-01-03")
    assert actual["end_event_date"][0] == pd.Timestamp("2020-01-03")
