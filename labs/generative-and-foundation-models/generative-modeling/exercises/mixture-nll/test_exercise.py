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


mixture_nll = load().mixture_nll


def direct_nll(x, centers, sd, w):
    p = sum(wk * multivariate_normal(mk, sd**2 * np.eye(len(mk))).pdf(x) for wk, mk in zip(w, centers))
    return -np.mean(np.log(p))


def test_single_gaussian_1d():
    # 표준정규분포에서 x = 0의 NLL은 0.5 log(2 pi)
    got = mixture_nll(np.zeros((1, 1)), np.zeros((1, 1)), 1.0)
    assert np.isclose(got, 0.5 * np.log(2 * np.pi)), f"N(0,1)에서 x=0의 NLL은 0.919입니다. 현재 {got}"


def test_matches_direct_computation_2d():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 2))
    x = rng.normal(size=(50, 2))
    w = np.array([0.2, 0.5, 0.3])
    got = mixture_nll(x, centers, 0.7, w)
    assert np.isclose(got, direct_nll(x, centers, 0.7, w)), "가중치가 있는 2차원 혼합의 NLL이 직접 계산과 다릅니다"


def test_uniform_weights_by_default_and_normalization():
    rng = np.random.default_rng(1)
    centers, x = rng.normal(size=(4, 3)), rng.normal(size=(20, 3))
    a = mixture_nll(x, centers, 0.5)
    b = mixture_nll(x, centers, 0.5, np.ones(4))          # 합이 1이 아닌 가중치는 정규화
    c = direct_nll(x, centers, 0.5, np.full(4, 0.25))
    assert np.isclose(a, c) and np.isclose(b, c), "weights=None은 균등 가중치, 합이 1이 아닌 가중치는 정규화합니다"


def test_far_points_do_not_underflow():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    x = np.array([[50.0, 0.0]])                           # 표준편차 0.1의 수백 배 떨어진 점
    got = mixture_nll(x, centers, 0.1)
    # 가까운 성분(1, 0)이 지배한다: -log(0.5) + 49^2/(2*0.01) + log(2 pi 0.01)
    expect = -np.log(0.5) + 49.0**2 / (2 * 0.01) + np.log(2 * np.pi * 0.01)
    assert np.isfinite(got), "멀리 떨어진 점에서 NLL이 inf가 되었습니다. logsumexp를 쓰세요"
    assert np.isclose(got, expect, rtol=1e-9), f"기대값 {expect:.3f}, 현재 {got:.3f}"


def test_returns_scalar_float():
    got = mixture_nll(np.zeros((3, 2)), np.zeros((1, 2)), 1.0)
    assert np.ndim(got) == 0, "평균 NLL 스칼라 하나를 반환합니다"
