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


tikhonov_svd = load().tikhonov_svd


def blur_problem(n=40, seed=0):
    s = np.linspace(0, 1, n)
    A = np.exp(-0.5 * (s[:, None] - s[None, :]) ** 2 / 0.05**2)
    A /= A.sum(axis=1, keepdims=True)
    x_true = (np.abs(s - 0.5) < 0.2).astype(float)
    y = A @ x_true + 0.01 * np.random.default_rng(seed).normal(size=n)
    return A, y


def test_matches_normal_equations():
    rng = np.random.default_rng(1)
    for m, n in [(30, 20), (20, 30), (25, 25)]:
        A, y = rng.normal(size=(m, n)), rng.normal(size=m)
        for lam in [1e-3, 0.1, 10.0]:
            x, f = tikhonov_svd(A, y, lam)
            x_ref = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ y)
            assert np.shape(x) == (n,), f"x의 shape은 (n,) = ({n},)이어야 합니다. 현재 {np.shape(x)}"
            assert np.shape(f) == (min(m, n),), f"f의 shape은 (min(m, n),)이어야 합니다. 현재 {np.shape(f)}"
            assert np.allclose(x, x_ref), "정규방정식의 해와 다릅니다"


def test_filter_factors():
    A, y = blur_problem()
    _, f = tikhonov_svd(A, y, 1e-4)
    assert np.all((f >= 0) & (f <= 1)), "필터 인자는 0과 1 사이여야 합니다"
    assert np.all(np.diff(f) <= 1e-12), "특이값이 큰 순서로 필터 인자도 커야(1에 가까워야) 합니다"
    assert f[0] > 0.99 and f[-1] < 1e-3, "큰 특이값은 거의 통과, 작은 특이값은 거의 차단해야 합니다"


def test_tradeoff_monotone():
    A, y = blur_problem()
    lams = [1e-6, 1e-4, 1e-2, 1.0]
    xs = [tikhonov_svd(A, y, lam)[0] for lam in lams]
    sol = [np.linalg.norm(x) for x in xs]
    res = [np.linalg.norm(A @ x - y) for x in xs]
    assert all(a > b for a, b in zip(sol, sol[1:])), "lam이 커지면 해 노름은 줄어야 합니다"
    assert all(a < b for a, b in zip(res, res[1:])), "lam이 커지면 잔차 노름은 늘어야 합니다"


def test_singular_matrix_finite():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])               # rank 1, 특이값 하나가 0
    x, f = tikhonov_svd(A, np.array([2.0, 2.0]), 0.1)
    assert np.all(np.isfinite(x)), "특이값이 0이어도 해가 유한해야 합니다"
    assert np.isclose(f[-1], 0.0), "특이값 0의 필터 인자는 0이어야 합니다"
    assert np.allclose(x, np.linalg.solve(A.T @ A + 0.1 * np.eye(2), A.T @ np.array([2.0, 2.0])))
