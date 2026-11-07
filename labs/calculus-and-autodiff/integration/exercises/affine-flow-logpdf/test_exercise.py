import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


affine_flow_logpdf = load().affine_flow_logpdf


def test_matches_gaussian_logpdf():
    rng = np.random.default_rng(0)
    for d in (1, 2, 4):
        A = rng.normal(size=(d, d)) + 2 * np.eye(d)
        b = rng.normal(size=d)
        Y = rng.normal(size=(7, d)) * 2
        out = affine_flow_logpdf(Y, A, b)
        assert np.shape(out) == (7,), f"반환 shape은 (N,) = (7,)이어야 합니다. 현재 {np.shape(out)}"
        ref = multivariate_normal(mean=b, cov=A @ A.T).logpdf(Y)
        assert np.allclose(out, np.atleast_1d(ref)), f"d={d}에서 평균 b, 공분산 AA^T인 정규분포의 로그밀도와 다릅니다"


def test_negative_determinant():
    A = np.array([[0.0, 2.0], [1.0, 0.0]])           # det = -2 (두 축을 바꾸는 반사 포함)
    b = np.array([0.5, -1.0])
    Y = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
    out = affine_flow_logpdf(Y, A, b)
    assert np.all(np.isfinite(out)), "det A < 0에서 nan이 나왔습니다. log|det A|의 절댓값을 확인하세요"
    ref = multivariate_normal(mean=b, cov=A @ A.T).logpdf(Y)
    assert np.allclose(out, ref)


def test_integrates_to_one():
    A, b = np.array([[1.5, 0.0], [0.7, 0.5]]), np.array([1.0, -0.5])
    g = np.linspace(-9, 11, 401)
    dx = g[1] - g[0]
    G = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    mass = np.exp(affine_flow_logpdf(G, A, b)).sum() * dx**2
    assert abs(mass - 1.0) < 1e-3, f"밀도의 전체 적분이 1이어야 합니다. 현재 {mass:.4f}. Jacobian 항을 빠뜨리지 않았는지 확인하세요"
