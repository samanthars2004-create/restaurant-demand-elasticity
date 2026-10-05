"""
Small, dependency-light econometrics helpers (numpy only).

Implements OLS and two-stage least squares (2SLS) with Newey-West
heteroskedasticity- and autocorrelation-consistent (HAC) standard errors,
plus a delta-method standard error for long-run elasticities.

Written without statsmodels so the project runs with only numpy/pandas/
scipy/matplotlib installed. tests/test_econometrics.py checks these
estimators against simulated data with known parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb bandwidth: floor(4 * (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def _hac_meat(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West 'meat' matrix with Bartlett kernel weights."""
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    return S


@dataclass
class RegResult:
    name: str
    params: pd.Series
    bse: pd.Series
    nobs: int
    r2: float
    lags: int
    resid: np.ndarray = field(repr=False)
    cov: pd.DataFrame = field(repr=False)
    first_stage_F: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        dof = self.nobs - len(self.params)
        return pd.Series(2 * stats.t.sf(np.abs(self.tvalues), dof), index=self.params.index)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.params, "se": self.bse, "t": self.tvalues, "p": self.pvalues}
        )


def ols(y: pd.Series, X: pd.DataFrame, name: str = "OLS", lags: int | None = None) -> RegResult:
    """OLS with Newey-West HAC standard errors."""
    yv = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    n, k = Xv.shape
    lags = newey_west_lags(n) if lags is None else lags
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ yv
    u = yv - Xv @ b
    V = XtX_inv @ _hac_meat(Xv, u, lags) @ XtX_inv * n / (n - k)
    r2 = 1 - (u @ u) / np.sum((yv - yv.mean()) ** 2)
    return RegResult(
        name=name,
        params=pd.Series(b, index=X.columns),
        bse=pd.Series(np.sqrt(np.diag(V)), index=X.columns),
        nobs=n,
        r2=float(r2),
        lags=lags,
        resid=u,
        cov=pd.DataFrame(V, index=X.columns, columns=X.columns),
    )


def iv2sls(
    y: pd.Series,
    exog: pd.DataFrame,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    name: str = "2SLS",
    lags: int | None = None,
) -> RegResult:
    """
    Two-stage least squares with Newey-West HAC standard errors.

    exog        : included exogenous regressors (incl. constant)
    endog       : endogenous regressor(s)
    instruments : excluded instruments
    """
    X = pd.concat([endog, exog], axis=1)
    Z = pd.concat([instruments, exog], axis=1)
    yv, Xv, Zv = (a.to_numpy(dtype=float) for a in (y, X, Z))
    n, k = Xv.shape
    lags = newey_west_lags(n) if lags is None else lags

    # First stage fitted values
    PZ = Zv @ np.linalg.pinv(Zv.T @ Zv) @ Zv.T
    Xhat = PZ @ Xv
    A_inv = np.linalg.pinv(Xhat.T @ Xv)
    b = A_inv @ Xhat.T @ yv
    u = yv - Xv @ b  # structural residuals use actual X
    V = A_inv @ _hac_meat(Xhat, u, lags) @ A_inv.T * n / (n - k)
    r2 = 1 - (u @ u) / np.sum((yv - yv.mean()) ** 2)

    # Robust first-stage F on excluded instruments (one endogenous variable)
    fs = ols(endog.iloc[:, 0], Z, name="first stage", lags=lags)
    ex = list(instruments.columns)
    bz = fs.params[ex].to_numpy()
    Vz = fs.cov.loc[ex, ex].to_numpy()
    F = float(bz @ np.linalg.pinv(Vz) @ bz) / len(ex)

    return RegResult(
        name=name,
        params=pd.Series(b, index=X.columns),
        bse=pd.Series(np.sqrt(np.diag(V)), index=X.columns),
        nobs=n,
        r2=float(r2),
        lags=lags,
        resid=u,
        cov=pd.DataFrame(V, index=X.columns, columns=X.columns),
        first_stage_F=F,
        extra={"first_stage": fs},
    )


def long_run(res: RegResult, coef: str, lag_coef: str) -> tuple[float, float]:
    """
    Long-run effect in a partial-adjustment model: coef / (1 - lag_coef),
    with a delta-method standard error.
    """
    b = res.params[coef]
    lam = res.params[lag_coef]
    est = b / (1 - lam)
    grad = np.array([1 / (1 - lam), b / (1 - lam) ** 2])
    V = res.cov.loc[[coef, lag_coef], [coef, lag_coef]].to_numpy()
    se = float(np.sqrt(grad @ V @ grad))
    return float(est), se
