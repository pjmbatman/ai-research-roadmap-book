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


abc_accept = load().abc_accept


def toy_table(N=4000, seed=0):
    """theta ~ U(0, 1), 통계 1 = theta + 잡음(정보 있음), 통계 2 = 잡음(정보 없음)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 1, N)
    S = np.stack([theta + 0.05 * rng.normal(size=N), rng.normal(size=N)], axis=1)   # (N, 2)
    return theta, S


def distances(S, s_obs):
    return np.sqrt((((S - s_obs) / S.std(0)) ** 2).sum(1))


def test_number_accepted():
    theta, S = toy_table()
    acc = abc_accept(theta, S, np.array([0.3, 0.0]), 0.05)
    assert abs(len(acc) - 200) <= 2, f"q=0.05, N=4000이면 약 200개를 받아들여야 합니다. 현재 {len(acc)}개"


def test_accepts_the_closest():
    theta, S = toy_table()
    s_obs = np.array([0.7, 0.5])
    acc = abc_accept(theta, S, s_obs, 0.02)
    d = distances(S, s_obs)
    is_acc = np.isin(theta, acc)
    assert is_acc.sum() == len(acc), "반환값은 theta_sim의 원소여야 합니다"
    assert d[is_acc].max() <= d[~is_acc].min(), "받아들인 표본은 거부한 표본보다 모두 가까워야 합니다"


def test_standardization():
    theta, S = toy_table()
    s_obs = np.array([0.4, -0.2])
    unit = np.array([1.0, 1000.0])                            # 둘째 통계의 단위만 바꿈
    a = np.sort(abc_accept(theta, S, s_obs, 0.05))
    b = np.sort(abc_accept(theta, S * unit, s_obs * unit, 0.05))
    assert np.array_equal(a, b), "통계의 단위(척도)를 바꿔도 결과가 같아야 합니다. 열마다 표준편차로 나누었나요?"


def test_informative_summary_concentrates():
    theta, S = toy_table(N=20000, seed=1)
    acc = abc_accept(theta, S[:, :1], np.array([0.3]), 0.01)  # 정보 있는 통계만
    assert abs(acc.mean() - 0.3) < 0.03, f"받아들인 θ의 평균이 0.3 근처여야 합니다. 현재 {acc.mean():.3f}"
    assert acc.std() < 0.08, f"정보 있는 통계를 쓰면 사후분포가 좁아야 합니다. 현재 표준편차 {acc.std():.3f}"
