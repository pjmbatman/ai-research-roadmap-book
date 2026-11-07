import numpy as np


def ols_fit(X, y):
    """최소제곱 계수, 표준오차, 잔차 분산 {"beta", "se", "sigma2"}. sigma2는 RSS / (n - p)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)           # (p, p)
    beta = XtX_inv @ X.T @ y                   # (p,)
    resid = y - X @ beta
    sigma2 = float(resid @ resid / (n - p))
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    return {"beta": beta, "se": se, "sigma2": sigma2}
