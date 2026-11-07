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


pca = load().pca


def make_data(seed, N=300, offset=(0.0, 0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N, 4)) * [3.0, 2.0, 1.0, 0.3]     # 분산이 서로 다른 잠재 축
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0]            # 무작위 회전
    return z @ R.T + np.asarray(offset)


def reference(X, k):
    w, Q = np.linalg.eigh(np.cov(X, rowvar=False))
    return w[::-1][:k], Q[:, ::-1][:, :k].T                  # (k,), (k, D)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_matches_covariance_eigendecomposition(k):
    X = make_data(0)
    out = pca(X, k)
    var_ref, comp_ref = reference(X, k)
    comps = np.asarray(out["components"])
    assert comps.shape == (k, 4), f"components의 shape은 (k, D) = {(k, 4)}이어야 합니다. 현재 {comps.shape}"
    assert np.allclose(out["explained_variance"], var_ref), "분산은 s^2 / (N - 1)입니다"
    cos = np.abs(np.sum(comps * comp_ref, axis=1))
    assert np.allclose(cos, 1), f"주성분 방향이 공분산 고유벡터와 다릅니다 |cos| = {cos}. Vt의 행을 썼는지 확인합니다"


def test_centering():
    X = make_data(1, offset=(50.0, -20.0, 10.0, 30.0))
    out = pca(X, 2)
    _, comp_ref = reference(X, 2)
    assert np.allclose(out["mean"], X.mean(axis=0)), "mean은 열 평균입니다"
    cos = np.abs(np.sum(np.asarray(out["components"]) * comp_ref, axis=1))
    assert np.allclose(cos, 1), "평균을 빼지 않으면 첫 방향이 평균 방향으로 끌려갑니다"


def test_scores():
    X = make_data(2)
    out = pca(X, 3)
    S = np.asarray(out["scores"])
    assert S.shape == (300, 3), f"scores의 shape은 (N, k) = (300, 3)이어야 합니다. 현재 {S.shape}"
    assert np.allclose(S, (X - X.mean(0)) @ np.asarray(out["components"]).T), "scores = (X - mean) @ components.T"
    C = np.cov(S, rowvar=False)
    assert np.allclose(np.diag(C), out["explained_variance"]), "점수의 분산이 explained_variance와 같아야 합니다"
    assert np.allclose(C - np.diag(np.diag(C)), 0, atol=1e-10), "서로 다른 주성분 점수는 상관이 없습니다"


def test_reconstruction():
    X = make_data(3)
    full = pca(X, 4)
    X_rec = np.asarray(full["scores"]) @ np.asarray(full["components"]) + full["mean"]
    assert np.allclose(X_rec, X), "k = D이면 재구성이 원래 데이터와 같습니다"
    two = pca(X, 2)
    X2 = np.asarray(two["scores"]) @ np.asarray(two["components"]) + two["mean"]
    err2 = np.sum((X - X2) ** 2) / (len(X) - 1)
    assert np.isclose(err2, np.sum(full["explained_variance"][2:])), "rank-2 재구성 오차 = 버린 주성분 분산의 합"
