import numpy as np


def dlqr(A, B, Q, R, tol=1e-10, max_iter=100_000):
    """이산 시간 무한 지평 LQR을 Riccati 반복으로 푼다.

    A: (n, n), B: (n, m), Q: (n, n) 상태 비용, R: (m, m) 입력 비용
    반환: K (m, n) 최적 이득(u = -K x), P (n, n) 가치함수 행렬(V(x) = x^T P x)
    """
    A, B, Q, R = (np.asarray(M, dtype=float) for M in (A, B, Q, R))
    P = Q.copy()
    for _ in range(max_iter):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)    # (m, n)
        P_new = Q + A.T @ P @ (A - B @ K)                    # (n, n)
        P_new = 0.5 * (P_new + P_new.T)                      # 대칭 유지
        if np.abs(P_new - P).max() < tol:
            P = P_new
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
