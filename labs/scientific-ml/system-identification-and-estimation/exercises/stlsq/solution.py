import numpy as np


def stlsq(Theta, dX, thresh, iters=10):
    """순차 임계 최소제곱으로 dX ≈ Theta @ Xi의 희소한 Xi를 구한다.

    Theta: (T, p) 후보 함수 라이브러리, dX: (T, d) 시간미분
    thresh: 이보다 절댓값이 작은 계수를 0으로 만든다
    반환: Xi (p, d)
    """
    Theta, dX = np.asarray(Theta, dtype=float), np.asarray(dX, dtype=float)
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]  # (p, d)
    for _ in range(iters):
        small = np.abs(Xi) < thresh
        Xi[small] = 0.0
        for k in range(dX.shape[1]):
            keep = ~small[:, k]
            if keep.any():
                Xi[keep, k] = np.linalg.lstsq(Theta[:, keep], dX[:, k], rcond=None)[0]
    return Xi
