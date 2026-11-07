import numpy as np


def stlsq(Theta, dX, thr=0.05, iters=10):
    """순차적 임계 최소제곱으로 희소한 계수 행렬을 찾는다.

    Theta: (T, p) 후보 함수 사전
    dX: (T, n) 상태의 시간 미분
    thr: 임계값. 절댓값이 이보다 작은 계수는 0으로 만든다
    iters: 자르기-다시 맞추기 반복 횟수
    반환: Xi (p, n)
    """
    Theta = np.asarray(Theta, dtype=float)
    dX = np.asarray(dX, dtype=float).reshape(len(Theta), -1)   # (T, n)
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]             # (p, n)
    for _ in range(iters):
        small = np.abs(Xi) < thr
        Xi[small] = 0.0
        for j in range(dX.shape[1]):
            keep = ~small[:, j]
            if keep.any():                                     # 남은 항만으로 다시 맞춘다
                Xi[keep, j] = np.linalg.lstsq(Theta[:, keep], dX[:, j], rcond=None)[0]
    return Xi
