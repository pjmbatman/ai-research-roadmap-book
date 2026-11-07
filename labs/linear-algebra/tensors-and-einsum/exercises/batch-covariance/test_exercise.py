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


batch_cov = load().batch_cov


@pytest.mark.parametrize("B,N,D", [(3, 20, 2), (5, 7, 4), (1, 50, 3)])
def test_matches_np_cov(B, N, D):
    rng = np.random.default_rng(B * 100 + N + D)
    X = rng.normal(size=(B, N, D))
    S = batch_cov(X)
    assert np.shape(S) == (B, D, D), f"shape은 (B, D, D) = {(B, D, D)}여야 합니다. 현재 {np.shape(S)}"
    for b in range(B):
        assert np.allclose(S[b], np.cov(X[b], rowvar=False)), f"batch 항목 {b}의 공분산이 np.cov와 다릅니다"


def test_centers_each_item_separately():
    # 센서마다 기준점(offset)이 크게 다르다. 전체 평균을 빼면 offset이 분산처럼 보인다.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 30, 2)) + np.array([0.0, 10.0, -20.0, 50.0])[:, None, None]
    S = batch_cov(X)
    assert np.all(np.diagonal(S, axis1=1, axis2=2) < 5), "평균은 batch 항목마다 샘플 축(axis=1)으로 빼야 합니다"


def test_uses_n_minus_one():
    X = np.array([[[0.0], [2.0]]])          # (1, 2, 1): 두 값 0, 2 → 표본분산 2
    assert np.allclose(batch_cov(X), [[[2.0]]]), "표본 공분산은 N이 아니라 N-1로 나눕니다"


def test_symmetric_psd():
    rng = np.random.default_rng(2)
    S = batch_cov(rng.normal(size=(6, 15, 5)))
    assert np.allclose(S, np.swapaxes(S, 1, 2)), "공분산 행렬은 대칭이어야 합니다"
    assert np.linalg.eigvalsh(S).min() > -1e-10, "공분산 행렬의 고윳값은 음수가 아니어야 합니다"
