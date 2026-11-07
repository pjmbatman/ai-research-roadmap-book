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


pairwise_sq_dists = load().pairwise_sq_dists


def loop_version(X, Y):
    S = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            S[i, j] = np.sum((X[i] - Y[j]) ** 2)
    return S


@pytest.mark.parametrize("N,M,D", [(5, 7, 3), (6, 6, 2), (4, 3, 1), (1, 5, 4)])
def test_matches_loop(N, M, D):
    rng = np.random.default_rng(N * 100 + M * 10 + D)
    X, Y = rng.normal(size=(N, D)), rng.normal(size=(M, D))
    S = pairwise_sq_dists(X, Y)
    assert np.shape(S) == (N, M), f"shape은 (N, M) = ({N}, {M})이어야 합니다. 현재 {np.shape(S)}"
    assert np.allclose(S, loop_version(X, Y)), "반복문 결과와 다릅니다"


def test_not_transposed():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])          # (2, 2)
    Y = np.array([[0.0, 3.0], [0.0, 0.0], [2.0, 0.0]])  # (3, 2)
    S = pairwise_sq_dists(X, Y)
    assert np.allclose(S, [[9.0, 0.0, 4.0], [10.0, 1.0, 1.0]]), f"S[i, j]는 X[i]와 Y[j]의 거리입니다. 현재\n{S}"


def test_nonnegative_on_duplicates():
    rng = np.random.default_rng(3)
    X = 1e4 + rng.normal(size=(50, 8))               # 큰 값: 전개식에서 소거 오차가 생기기 쉽다
    S = pairwise_sq_dists(X, X)
    assert np.all(S >= 0), f"음수 거리가 있습니다. 최솟값 {S.min()}"
    assert np.allclose(np.diag(S), 0, atol=1e-6)


def test_inputs_unchanged():
    rng = np.random.default_rng(4)
    X, Y = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    X0, Y0 = X.copy(), Y.copy()
    pairwise_sq_dists(X, Y)
    assert np.array_equal(X, X0) and np.array_equal(Y, Y0), "입력 배열을 바꾸면 안 됩니다"
