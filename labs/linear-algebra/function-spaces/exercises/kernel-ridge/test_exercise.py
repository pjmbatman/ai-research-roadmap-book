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


kernel_ridge_predict = load().kernel_ridge_predict


def data(seed=0, n=15):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)


def test_matches_closed_form():
    x, y = data()
    xt = np.linspace(-0.2, 1.2, 50)
    ell, lam = 0.15, 0.05
    k = lambda a, b: np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * ell**2))
    expected = k(xt, x) @ np.linalg.solve(k(x, x) + lam * np.eye(len(x)), y)
    pred = kernel_ridge_predict(x, y, xt, ell, lam)
    assert np.shape(pred) == (50,), f"반환 shape은 (M,) = (50,)이어야 합니다. 현재 {np.shape(pred)}"
    assert np.allclose(pred, expected), "닫힌 형태 해 k(x_test, X) (K + λI)^{-1} y와 다릅니다"


def test_small_lambda_interpolates():
    x, y = data(1, n=8)
    pred = kernel_ridge_predict(x, y, x, 0.1, 1e-10)
    assert np.allclose(pred, y, atol=1e-4), "λ → 0이면 학습 점을 지나가야 합니다"


def test_large_lambda_shrinks():
    x, y = data(2)
    pred = kernel_ridge_predict(x, y, x, 0.1, 1e6)
    assert np.abs(pred).max() < 1e-3, "λ가 아주 크면 RKHS 노름 벌점 때문에 f ≈ 0이어야 합니다"


def test_far_from_data_goes_to_zero():
    x, y = data(3)
    pred = kernel_ridge_predict(x, y, np.array([10.0, -10.0]), 0.1, 0.01)
    assert np.allclose(pred, 0.0, atol=1e-8), "학습 점에서 먼 곳에서는 커널 함수들이 모두 0에 가까우므로 예측도 0에 가깝습니다"


def test_linear_in_y():
    x, y1 = data(4)
    _, y2 = data(5)
    xt = np.linspace(0, 1, 20)
    f = lambda y: kernel_ridge_predict(x, y, xt, 0.2, 0.1)
    assert np.allclose(f(2 * y1 - 3 * y2), 2 * f(y1) - 3 * f(y2)), "예측은 관측 y에 대해 선형이어야 합니다"
