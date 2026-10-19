import pandas as pd
import polars as pl

from era_pl import rdplot
import era_py.plots as pandas_plots
import era_pl.plots as polars_plots


def test_rdplot_returns_plotnine_object():
    df = pl.DataFrame(
        {
            "running": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "outcome": [1.0, 1.2, 1.1, 2.0, 2.1, 2.2],
        }
    )

    fig = rdplot(
        df,
        y="outcome",
        x="running",
        cutoff=0.0,
        y_label="Outcome",
        x_label="Running variable",
    )

    assert fig.__class__.__name__ == "ggplot"


def test_rdplot_accepts_boolean_outcome():
    df = pl.DataFrame(
        {
            "running": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "outcome": [False, False, True, True, True, True],
        }
    )

    fig = rdplot(
        df,
        y="outcome",
        x="running",
        cutoff=0.0,
        y_label="Outcome",
        x_label="Running variable",
        bins=2,
    )

    assert fig.__class__.__name__ == "ggplot"


def test_polars_rdplot_delegates_to_shared_pandas_implementation(monkeypatch):
    calls = {}

    def fake_rdplot(data, **kwargs):
        calls["data_type"] = type(data).__name__
        calls["kwargs"] = kwargs
        return "plot"

    monkeypatch.setattr(polars_plots, "_rdplot", fake_rdplot)

    result = polars_plots.rdplot(
        pl.DataFrame({"running": [-1.0, 1.0], "outcome": [0.0, 1.0]}),
        y="outcome",
        x="running",
        cutoff=0.0,
        y_label="Outcome",
        x_label="Running variable",
        bins=(1, 1),
    )

    assert result == "plot"
    assert calls["data_type"] == "DataFrame"
    assert calls["kwargs"]["bins"] == (1, 1)
    assert calls["kwargs"]["masspoints"] == "adjust"
    assert calls["kwargs"]["use_rdrobust"] is True


def test_polars_spline_smooth_delegates_to_shared_pandas_implementation(monkeypatch):
    calls = {}

    def fake_spline_smooth(data, **kwargs):
        calls["data_type"] = type(data).__name__
        calls["kwargs"] = kwargs
        return "smooth"

    monkeypatch.setattr(polars_plots, "_spline_smooth", fake_spline_smooth)

    result = polars_plots.spline_smooth(
        pl.DataFrame({"running": [-1.0, 1.0], "outcome": [0.0, 1.0]}),
        y="outcome",
        x="running",
        df=3,
        n=10,
    )

    assert result == "smooth"
    assert calls["data_type"] == "DataFrame"
    assert calls["kwargs"] == {"x": "running", "y": "outcome", "df": 3, "n": 10}


def test_rdplot_uses_rdrobust_returned_plot_data(monkeypatch):
    class FakeRdrobust:
        @staticmethod
        def rdplot(*args, **kwargs):
            class Result:
                vars_bins = pd.DataFrame(
                    {"rdplot_mean_bin": [-1.0, 1.0], "rdplot_mean_y": [0.25, 0.75]}
                )
                vars_poly = pd.DataFrame(
                    {"rdplot_x": [-1.0, 0.0, 0.0, 1.0], "rdplot_y": [0.2, 0.3, 0.7, 0.8]}
                )

            return Result()

    real_import_module = pandas_plots.importlib.import_module

    def fake_import_module(name):
        if name == "rdrobust":
            return FakeRdrobust
        return real_import_module(name)

    monkeypatch.setattr(pandas_plots.importlib, "import_module", fake_import_module)
    clean = pd.DataFrame({"running": [-1.0, 1.0], "outcome": [0.0, 1.0]})

    bins_df, line_df = pandas_plots._rdrobust_plot_data(
        clean,
        y="outcome",
        x="running",
        cutoff=0.0,
        degree=4,
        bins=None,
        binselect="esmv",
        masspoints="off",
    )

    assert bins_df["x"].tolist() == [-1.0, 1.0]
    assert bins_df["side"].tolist() == ["Left", "Right"]
    assert line_df["y"].tolist() == [0.2, 0.3, 0.7, 0.8]
