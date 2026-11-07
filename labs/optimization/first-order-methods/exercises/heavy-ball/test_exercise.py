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


heavy_ball = load().heavy_ball
LAM = np.array([1.0, 100.0])                    # mu = 1, L = 100
grad = lambda x: LAM * x


def steps_to(path, tol=1e-6):
    norms = np.linalg.norm(path, axis=1) / np.linalg.norm(path[0])
    hit = np.flatnonzero(norms < tol)
    return int(hit[0]) if hit.size else None


def test_shape_and_start():
    P = heavy_ball(grad, np.array([1.0, 1.0]), 0.01, 0.5, 7)
    assert np.shape(P) == (8, 2), f"반환 shape은 (steps + 1, n) = (8, 2)여야 합니다. 현재 {np.shape(P)}"
    assert np.allclose(P[0], [1.0, 1.0]), "첫 행은 시작점이어야 합니다"


def test_beta_zero_is_gd():
    x0 = np.array([1.0, -2.0])
    P = heavy_ball(grad, x0, 0.015, 0.0, 20)
    x = x0.copy()
    for _ in range(20):
        x = x - 0.015 * grad(x)
    assert np.allclose(P[-1], x), "beta = 0이면 경사하강과 같아야 합니다"


def test_first_two_steps():
    x0, lr, beta = np.array([1.0, 1.0]), 0.01, 0.9
    P = heavy_ball(grad, x0, lr, beta, 2)
    v1 = -lr * grad(x0)
    x1 = x0 + v1
    x2 = x1 + beta * v1 - lr * grad(x1)
    assert np.allclose(P[1], x1) and np.allclose(P[2], x2), "v는 0에서 시작하고, 이전 속도에 beta를 곱해 누적합니다"


def test_optimal_momentum_is_faster():
    mu, L = 1.0, 100.0
    kappa = L / mu
    x0 = np.array([1.0, 1.0])
    gd = heavy_ball(grad, x0, 2 / (L + mu), 0.0, 3000)
    hb = heavy_ball(grad, x0, 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2,
                    ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2, 3000)
    k_gd, k_hb = steps_to(gd), steps_to(hb)
    assert k_gd is not None and k_hb is not None
    assert k_hb * 5 <= k_gd, f"최적 momentum은 GD보다 훨씬 빨라야 합니다 (GD {k_gd} step, heavy ball {k_hb} step)"


def test_stability_boundary():
    L, beta = 100.0, 0.9
    x0 = np.array([0.3, 1.0])
    ok = heavy_ball(grad, x0, 0.99 * 2 * (1 + beta) / L, beta, 4000)
    with np.errstate(over="ignore", invalid="ignore"):
        bad = heavy_ball(grad, x0, 1.01 * 2 * (1 + beta) / L, beta, 4000)
        bad_norm = np.linalg.norm(bad[-1])
    assert np.linalg.norm(ok[-1]) < 1e-3, "lr < 2(1 + beta)/L 이면 수렴해야 합니다"
    assert not np.isfinite(bad_norm) or bad_norm > 1e3, "lr > 2(1 + beta)/L 이면 발산해야 합니다"
