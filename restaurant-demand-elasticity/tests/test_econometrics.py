"""
Sanity checks for src/econometrics.py using simulated data with known
parameters. Run:  python tests/test_econometrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from econometrics import iv2sls, long_run, ols  # noqa: E402


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = 1.0 - 0.3 * x + rng.normal(scale=0.5, size=n)
    X = pd.DataFrame({"const": 1.0, "x": x})
    r = ols(pd.Series(y), X)
    assert abs(r.params["x"] + 0.3) < 0.03, r.params
    assert abs(r.params["const"] - 1.0) < 0.03, r.params


def test_iv_fixes_simultaneity_bias():
    # Price is correlated with the demand shock (endogenous); cost shifter z is not.
    rng = np.random.default_rng(1)
    n = 20000
    z = rng.normal(size=n)
    e = rng.normal(size=n)
    p = 0.8 * z + 0.6 * e + rng.normal(scale=0.3, size=n)
    q = 2.0 - 0.25 * p + e
    exog = pd.DataFrame({"const": np.ones(n)})
    r_ols = ols(pd.Series(q), pd.concat([pd.DataFrame({"p": p}), exog], axis=1))
    r_iv = iv2sls(pd.Series(q), exog, pd.DataFrame({"p": p}), pd.DataFrame({"z": z}))
    assert abs(r_iv.params["p"] + 0.25) < 0.03, r_iv.params
    assert abs(r_ols.params["p"] + 0.25) > 0.15, "OLS should be biased here"
    assert r_iv.first_stage_F > 100


def test_long_run_delta_method():
    rng = np.random.default_rng(2)
    n = 20000
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] - 0.1 * x[t] + rng.normal(scale=0.1)
    df = pd.DataFrame({"y": y, "x": x})
    df["y_lag"] = df["y"].shift(1)
    df = df.dropna()
    X = pd.DataFrame({"const": 1.0, "x": df["x"], "y_lag": df["y_lag"]})
    r = ols(df["y"], X)
    lr, se = long_run(r, "x", "y_lag")
    assert abs(lr + 0.25) < 0.02, lr  # -0.1 / (1 - 0.6) = -0.25
    assert 0 < se < 0.02


if __name__ == "__main__":
    test_ols_recovers_coefficients()
    test_iv_fixes_simultaneity_bias()
    test_long_run_delta_method()
    print("All econometrics tests passed.")
