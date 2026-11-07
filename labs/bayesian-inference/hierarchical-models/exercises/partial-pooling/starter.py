import numpy as np


def partial_pool(ybar, se, mu, tau):
    """μ, τ를 알 때 θ_j의 사후평균과 사후 표준편차를 반환한다.

    ybar: (J,) 그룹 표본평균, se: (J,) 표준오차
    반환: mean (J,), sd (J,)
    """
    raise NotImplementedError("수축 계수 B_j로 사후평균과 표준편차를 계산하세요")


def marginal_loglik(ybar, se, mu, tau):
    """Σ_j log N(ybar_j | mu, tau² + se_j²)를 반환한다."""
    raise NotImplementedError("주변 로그우도를 계산하세요")


def eb_estimate(ybar, se, taus):
    """후보 taus에서 주변 로그우도를 최대화하는 (mu_hat, tau_hat)을 반환한다(μ는 정밀도 가중평균)."""
    raise NotImplementedError("τ 후보마다 μ̂과 주변 로그우도를 계산하세요")
