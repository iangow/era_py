from types import SimpleNamespace

from era_py.plots import plotnine_star


def test_plotnine_star_restores_top_level_namespace():
    module = SimpleNamespace(alpha=1, beta=2, _hidden=3)
    namespace = {"plotnine_star": plotnine_star, "module": module}

    exec(
        """
before = ("alpha" in globals(), "beta" in globals(), "_hidden" in globals())
with plotnine_star(module=module):
    inside = (alpha, beta, "alpha" in globals(), "beta" in globals())
after = ("alpha" in globals(), "beta" in globals(), "_hidden" in globals())
""",
        namespace,
    )

    assert namespace["before"] == (False, False, False)
    assert namespace["inside"] == (1, 2, True, True)
    assert namespace["after"] == (False, False, False)


def test_plotnine_star_restores_separate_exec_locals():
    module = SimpleNamespace(alpha=1)
    globals_ns = {"plotnine_star": plotnine_star, "module": module}
    locals_ns = {"alpha": "original"}

    exec(
        """
before = (alpha, "alpha" in globals(), "alpha" in locals())
with plotnine_star(module=module):
    inside = (alpha, globals()["alpha"], locals()["alpha"])
after = (alpha, "alpha" in globals(), "alpha" in locals())
""",
        globals_ns,
        locals_ns,
    )

    assert locals_ns["before"] == ("original", False, True)
    assert locals_ns["inside"] == (1, 1, 1)
    assert locals_ns["after"] == ("original", False, True)
