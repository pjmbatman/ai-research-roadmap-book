import numpy as np


def ols_fit(X, y):
    """최소제곱 계수, 표준오차, 잔차 분산 {"beta", "se", "sigma2"}. sigma2는 RSS / (n - p)."""
    raise NotImplementedError("β̂ = (XᵀX)⁻¹Xᵀy, σ̂² = RSS/(n-p), SE = sqrt(diag(σ̂²(XᵀX)⁻¹))")
