import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


random_shooting = load().random_shooting


def integrator(X, U):                        # x' = x + u, X (N, 1), U (N, 1)
    return X + U


def sq_cost(X_next, U):                      # 다음 상태의 제곱, (N,)
    return np.sum(X_next**2, axis=1)


def test_returns_first_action_in_bounds():
    rng = np.random.default_rng(0)
    u = random_shooting(np.array([0.3]), integrator, sq_cost, H=5, N=200,
                        u_low=np.array([-1.0]), u_high=np.array([1.0]), rng=rng)
    assert np.shape(u) == (1,), f"첫 입력 하나의 shape은 (m,) = (1,)이어야 합니다. 현재 {np.shape(u)}"
    assert -1.0 <= u[0] <= 1.0, "입력은 [u_low, u_high] 안에서 뽑아야 합니다"


def test_unconstrained_optimum_one_step():
    rng = np.random.default_rng(1)
    u = random_shooting(np.array([0.5]), integrator, sq_cost, H=1, N=4000,
                        u_low=np.array([-1.0]), u_high=np.array([1.0]), rng=rng)
    assert abs(u[0] + 0.5) < 0.01, "x = 0.5, x' = x + u의 제곱을 최소화하는 입력은 -0.5입니다"


def test_constraint_active():
    rng = np.random.default_rng(2)
    u = random_shooting(np.array([3.0]), integrator, sq_cost, H=1, N=4000,
                        u_low=np.array([-1.0]), u_high=np.array([1.0]), rng=rng)
    assert u[0] < -0.99, "원점에서 멀면 한계 -1 근처의 입력을 골라야 합니다"


def test_horizon_sums_costs_along_rollout():
    # 지연이 있는 시스템: 상태 (위치, 다음에 적용될 입력). 지금 입력은 다음 step에야 위치를 바꾼다.
    def delayed(X, U):
        return np.stack([X[:, 0] + X[:, 1], U[:, 0]], axis=1)

    def pos_cost(X_next, U):
        return X_next[:, 0] ** 2

    rng = np.random.default_rng(3)
    x = np.array([1.0, 0.0])
    u = random_shooting(x, delayed, pos_cost, H=3, N=3000,
                        u_low=np.array([-1.0]), u_high=np.array([1.0]), rng=rng)
    assert u[0] < -0.8, "지평이 1보다 길어야 지연된 효과를 볼 수 있습니다. 지평 동안의 비용을 모두 더하세요"


def test_closed_loop_receding_horizon():
    rng = np.random.default_rng(4)
    x, traj = np.array([2.5]), []
    for _ in range(12):
        u = random_shooting(x, integrator, sq_cost, H=2, N=2000,
                            u_low=np.array([-1.0]), u_high=np.array([1.0]), rng=rng)
        x = integrator(x[None], u[None])[0]
        traj.append(abs(x[0]))
    assert max(traj[-5:]) < 0.2, "첫 입력만 적용하고 다시 계획하면 원점 근처에 머물러야 합니다"
