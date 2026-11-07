"""7.5 실습: 거듭제곱법으로 선형 동역학의 장기 거동을 예측하고 시뮬레이션과 비교한다.

1부: 연령 구조 개체군 모델(Leslie 행렬). 지배 고윳값 = 장기 성장률, 그 고유벡터 = 안정 연령 분포.
2부: 선형 RNN의 역전파에서 gradient 크기가 거슬러 간 step 수 K에 따라 어떻게 변하는지 스펙트럼 반지름으로 예측한다.

실행: uv run --project labs python labs/linear-algebra/eigen-and-symmetric/long_term.py
"""
import numpy as np


def power_iteration(A, num_iters=200, seed=0):
    """지배 고윳값과 고유벡터(단위 노름)를 거듭제곱법으로 구한다."""
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(num_iters):
        w = A @ v
        v = w / np.linalg.norm(w)
    return v @ A @ v / (v @ v), v


def leslie():
    # 세 연령 계층(0–1세, 1–2세, 2–3세). 첫 행 = 계층별 출산율, 아래 대각 = 다음 계층으로의 생존율
    L = np.array([[0.0, 1.5, 1.2],
                  [0.6, 0.0, 0.0],
                  [0.0, 0.4, 0.0]])                      # (3, 3)
    lam_pi, v_pi = power_iteration(L)
    lam_all = np.linalg.eigvals(L)
    print("[1부] Leslie 행렬")
    print("  numpy 고윳값       :", np.round(lam_all, 4))
    print(f"  거듭제곱법 지배 고윳값: {lam_pi:.4f}")
    v_pi = v_pi / v_pi.sum()                             # 합이 1인 연령 분포로
    print("  안정 연령 분포(예측):", np.round(v_pi, 4))

    n = np.array([100.0, 0.0, 0.0])                      # 0세 100마리로 시작
    for t in range(1, 41):
        n_next = L @ n
        if t in (1, 2, 5, 10, 40):
            print(f"  t = {t:2d}  총 개체 수 {n_next.sum():9.1f}  성장률 {n_next.sum() / n.sum():.4f}  "
                  f"연령 분포 {np.round(n_next / n_next.sum(), 4)}")
        n = n_next


def rnn_gradient():
    print("[2부] 선형 RNN h_t = W h_(t-1)의 gradient 크기 |(W^T)^K g|")
    rng = np.random.default_rng(1)
    W0 = rng.normal(size=(32, 32)) / np.sqrt(32)         # (32, 32)
    rho0 = max(abs(np.linalg.eigvals(W0)))
    g = rng.normal(size=32)
    g /= np.linalg.norm(g)
    for target in (0.9, 1.0, 1.1):
        W = W0 * (target / rho0)                         # 스펙트럼 반지름을 target으로
        sizes = []
        v = g.copy()
        for t in range(1, 101):
            v = W.T @ v
            if t in (10, 50, 100):
                sizes.append(np.linalg.norm(v))
        print(f"  rho = {target:.1f}  K = 10: {sizes[0]:9.2e}  K = 50: {sizes[1]:9.2e}  K = 100: {sizes[2]:9.2e}")


if __name__ == "__main__":
    leslie()
    rnn_gradient()
