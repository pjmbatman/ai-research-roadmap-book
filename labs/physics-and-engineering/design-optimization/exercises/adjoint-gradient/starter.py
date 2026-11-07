import numpy as np


def adjoint_gradient(k, f, c, h):
    """1차원 정상 열전도 K(k) T = f에서 J = c^T T와 dJ/dk를 adjoint 방법으로 계산한다.

    k: (n,) 요소 열전도도, f: (n,) 절점 열부하, c: (n,) 목적 가중치, h: 요소 길이
    반환: J (float), grad (n,)
    """
    raise NotImplementedError("K를 만들고, 해석 1회와 adjoint 해석 1회로 기울기를 구하세요")
