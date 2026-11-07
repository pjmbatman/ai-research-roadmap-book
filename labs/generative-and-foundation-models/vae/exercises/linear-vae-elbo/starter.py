import numpy as np


def elbo_terms(x, w, b, sigma, mu_q, s_q):
    """선형 가우시안 VAE에서 (재구성 항 E_q[log p(x|z)], KL(q(z|x) || N(0, 1)))을 닫힌 꼴로 반환한다.

    x, w, b: (d,), sigma: decoder 표준편차, mu_q, s_q: encoder 평균·표준편차(스칼라)
    """
    raise NotImplementedError("재구성 항과 KL 항을 계산하세요")
