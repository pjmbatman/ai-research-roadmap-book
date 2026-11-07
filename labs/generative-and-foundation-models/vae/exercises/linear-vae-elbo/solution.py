import numpy as np


def elbo_terms(x, w, b, sigma, mu_q, s_q):
    """선형 가우시안 VAE에서 (재구성 항 E_q[log p(x|z)], KL(q(z|x) || N(0, 1)))을 닫힌 꼴로 반환한다.

    x, w, b: (d,), sigma: decoder 표준편차, mu_q, s_q: encoder 평균·표준편차(스칼라)
    """
    x, w, b = (np.asarray(v, dtype=float) for v in (x, w, b))
    d = x.size
    r = x - b - w * mu_q
    sq_err = r @ r + (w @ w) * s_q**2                 # E_q ||x - b - w z||^2
    rec = -0.5 * d * np.log(2 * np.pi * sigma**2) - sq_err / (2 * sigma**2)
    kl = 0.5 * (mu_q**2 + s_q**2 - 1 - np.log(s_q**2))
    return float(rec), float(kl)
