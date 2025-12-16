import pandas as pd
import numpy as np
from era_py import ols_dropcollinear

def test_ols_dropcollinear_runs():
    df = pd.DataFrame({"y":[1,2,3,4], "x":[1,2,3,4], "g":[1,1,2,2]})
    res = ols_dropcollinear(df, "y ~ x + C(g)")
    assert hasattr(res, "params")
    assert hasattr(res, "dropped_terms")