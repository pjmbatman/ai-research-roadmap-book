import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


low_rank_approx = load().low_rank_approx


@pytest.mark.parametrize("m,n,k", [(8, 5, 1), (8, 5, 3), (6, 10, 2)])
def test_rank_and_error(m, n, k):
    rng = np.random.default_rng(m * 100 + n * 10 + k)
    A = rng.normal(size=(m, n))
    A_k, err = low_rank_approx(A, k)
    A_k = np.asarray(A_k)
    assert A_k.shape == A.shape, f"A_k의 shape은 A와 같아야 합니다. 현재 {A_k.shape}"
    assert np.linalg.matrix_rank(A_k) == k, f"A_k의 rank는 {k}여야 합니다"
    assert np.isclose(err, np.linalg.norm(A - A_k)), f"err {err}가 실제 오차 {np.linalg.norm(A - A_k)}와 다릅니다"
    s = np.linalg.svd(A, compute_uv=False)
    assert np.isclose(err, np.sqrt(np.sum(s[k:] ** 2))), "err는 버린 특이값의 제곱합의 제곱근입니다"


def test_beats_other_rank_k():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 12)) @ np.diag(np.linspace(3, 0.1, 12))
    A_k, err = low_rank_approx(A, 3)
    for _ in range(20):
        G = np.linalg.qr(rng.normal(size=(20, 3)))[0]
        assert err <= np.linalg.norm(A - G @ (G.T @ A)) + 1e-12, "최적 근사는 다른 rank-3 근사보다 오차가 작아야 합니다"


def test_exact_low_rank():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(10, 2)) @ rng.normal(size=(2, 7))       # rank 2
    A_k, err = low_rank_approx(A, 2)
    assert np.allclose(A_k, A) and err < 1e-10, "rank 2 행렬의 rank-2 근사는 자기 자신입니다"


def test_edge_cases():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 3))
    A0, e0 = low_rank_approx(A, 0)
    assert np.allclose(A0, 0) and np.isclose(e0, np.linalg.norm(A)), "k = 0이면 영행렬과 |A|_F"
    Af, ef = low_rank_approx(A, 5)
    assert np.allclose(Af, A) and np.isclose(ef, 0), "k >= min(m, n)이면 A 자신과 오차 0"


def test_row_vs_column_of_vt():
    # Vt의 열을 v_i로 잘못 쓰면 이 비대칭 예에서 틀린다
    A = np.array([[3.0, 0.0], [4.0, 5.0]])
    A1, err = low_rank_approx(A, 1)
    assert np.isclose(err, np.sqrt(5.0)), "특이값은 3√5, √5이므로 rank-1 오차는 √5입니다"
    assert np.isclose(np.linalg.norm(A - A1), np.sqrt(5.0)), "Vt의 행(Vt[i])이 v_i^T입니다"
