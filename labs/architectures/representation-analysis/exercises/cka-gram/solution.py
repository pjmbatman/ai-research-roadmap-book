import numpy as np


def cka_gram(K, L):
    """두 Gram 행렬로 CKA를 계산한다.

    K, L: (N, N) 같은 입력 N개에 대한 표본 간 유사도 행렬
    반환: float, HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))
    """
    K = np.asarray(K, dtype=float)
    L = np.asarray(L, dtype=float)

    def center(G):                                      # C G C (N, N)
        return G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()

    Kc, Lc = center(K), center(L)
    hsic = lambda X, Y: (X * Y.T).sum()                 # tr(X Y)
    return float(hsic(Kc, Lc) / np.sqrt(hsic(Kc, Kc) * hsic(Lc, Lc)))
