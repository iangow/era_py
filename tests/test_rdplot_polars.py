import polars as pl

from era_pl import rdplot


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
