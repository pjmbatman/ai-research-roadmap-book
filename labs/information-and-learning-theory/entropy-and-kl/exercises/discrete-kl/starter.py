import numpy as np


def cross_entropy(p, q):
    """H(p, q) = -Σ p log q [nat]. p_k = 0인 항은 0, p_k > 0이고 q_k = 0이면 inf.

    p, q: (K,) 확률 벡터
    """
    raise NotImplementedError("교차엔트로피를 계산하세요")


def kl_divergence(p, q):
    """KL(p || q) = Σ p log(p / q) [nat]. p_k = 0인 항은 0, p_k > 0이고 q_k = 0이면 inf.

    p, q: (K,) 확률 벡터
    """
    raise NotImplementedError("KL divergence를 계산하세요")
