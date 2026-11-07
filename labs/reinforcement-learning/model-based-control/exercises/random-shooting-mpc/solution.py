import numpy as np


def random_shooting(x, model, stage_cost, H, N, u_low, u_high, rng):
    """랜덤 슈팅 MPC의 한 step: 후보 입력열 N개를 모형으로 굴려 가장 좋은 열의 첫 입력을 반환한다.

    x: (n,) 현재 상태, model(X (N, n), U (N, m)) -> (N, n) 다음 상태,
    stage_cost(X_next (N, n), U (N, m)) -> (N,) 한 step 비용, H: 지평, N: 후보 수,
    u_low, u_high: (m,) 입력 한계, rng: np.random.Generator
    반환: (m,) 첫 입력
    """
    u_low, u_high = np.asarray(u_low, dtype=float), np.asarray(u_high, dtype=float)
    U = rng.uniform(u_low, u_high, size=(N, H, u_low.size))   # (N, H, m)
    X = np.tile(np.asarray(x, dtype=float), (N, 1))           # (N, n)
    J = np.zeros(N)
    for h in range(H):
        X = model(X, U[:, h])
        J += stage_cost(X, U[:, h])
    return U[np.argmin(J), 0]
