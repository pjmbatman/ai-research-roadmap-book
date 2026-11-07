import numpy as np


def dlqr(A, B, Q, R, tol=1e-10, max_iter=100_000):
    """이산 시간 무한 지평 LQR을 Riccati 반복으로 푼다.

    A: (n, n), B: (n, m), Q: (n, n) 상태 비용, R: (m, m) 입력 비용
    반환: K (m, n) 최적 이득(u = -K x), P (n, n) 가치함수 행렬(V(x) = x^T P x)
    """
    raise NotImplementedError("P = Q에서 출발해 K와 P를 번갈아 갱신하세요")
