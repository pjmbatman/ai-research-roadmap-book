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



top_eig = load().top_hessian_eigenvalue


def test_diagonal():
    D = np.diag([1.0, 7.0, 3.0])
    lam, v = top_eig(lambda u: D @ u, 3)
    assert np.isclose(lam, 7.0, rtol=1e-6), f"최대 고윳값은 7이어야 합니다. 현재 {lam}"
    assert np.isclose(abs(v[1]), 1.0, atol=1e-4), "고유벡터는 두 번째 좌표축 방향이어야 합니다"


def test_random_psd_matches_eigvalsh():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 30))
    Hm = A @ A.T / 30                                       # (30, 30) 대칭 양의 준정부호
    lam, v = top_eig(lambda u: Hm @ u, 30, iters=300)
    assert np.isclose(lam, np.linalg.eigvalsh(Hm)[-1], rtol=1e-4)
    assert np.isclose(np.linalg.norm(v), 1.0), "반환하는 벡터는 단위 벡터여야 합니다"
    assert np.allclose(Hm @ v, lam * v, atol=1e-3), "H v = lambda v를 만족해야 합니다"


def test_negative_dominant_eigenvalue():
    D = np.diag([-5.0, 2.0, 1.0])                           # 절댓값이 가장 큰 고윳값이 음수(안장점)
    lam, _ = top_eig(lambda u: D @ u, 3, iters=200)
    assert np.isclose(lam, -5.0, rtol=1e-6), "절댓값이 가장 큰 고윳값(부호 포함)을 반환해야 합니다"


def test_uses_only_matrix_vector_products():
    calls = []
    D = np.diag(np.arange(1.0, 11.0))
    def hvp(u):
        calls.append(np.shape(u))
        return D @ u
    top_eig(hvp, 10, iters=50)
    assert calls and all(c == (10,) for c in calls), "hvp에는 길이 n인 벡터만 넘깁니다"
