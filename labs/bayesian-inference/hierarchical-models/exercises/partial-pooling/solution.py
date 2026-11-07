import numpy as np


def partial_pool(ybar, se, mu, tau):
    """μ, τ를 알 때 θ_j의 사후평균과 사후 표준편차를 반환한다.

    ybar: (J,) 그룹 표본평균, se: (J,) 표준오차
    반환: mean (J,), sd (J,)
    """
    ybar, se = np.asarray(ybar, dtype=float), np.asarray(se, dtype=float)
    B = se**2 / (se**2 + tau**2)                          # 수축 계수, (J,)
    mean = B * mu + (1 - B) * ybar
    sd = 1 / np.sqrt(1 / se**2 + 1 / tau**2)
    return mean, sd


def marginal_loglik(ybar, se, mu, tau):
    """Σ_j log N(ybar_j | mu, tau² + se_j²)를 반환한다."""
    ybar, se = np.asarray(ybar, dtype=float), np.asarray(se, dtype=float)
    v = tau**2 + se**2
    return float(np.sum(-0.5 * np.log(2 * np.pi * v) - (ybar - mu) ** 2 / (2 * v)))


def eb_estimate(ybar, se, taus):
    """후보 taus에서 주변 로그우도를 최대화하는 (mu_hat, tau_hat)을 반환한다(μ는 정밀도 가중평균)."""
    ybar, se = np.asarray(ybar, dtype=float), np.asarray(se, dtype=float)
    best = (-np.inf, None, None)
    for t in np.asarray(taus, dtype=float):
        w = 1 / (t**2 + se**2)
        mu = np.sum(w * ybar) / np.sum(w)
        ll = marginal_loglik(ybar, se, mu, t)
        if ll > best[0]:
            best = (ll, mu, t)
    return best[1], best[2]
