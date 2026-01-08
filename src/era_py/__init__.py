__version__ = "0.0.6"

from .models import ols_dropcollinear
from .data import load_farr_rda
from .tables import modelsummary
from .plots import spline_smooth

__all__ = [
    "ols_dropcollinear",
    "load_farr_rda",
    "modelsummary",
    "spline_smooth",
    "__version__",
]
