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


mod = load()


def confounded(seed, n=20000):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * Z))).astype(int)    # Z가 크면 처치를 더 받음
    y = 2.0 * Z - 1.0 * t + rng.normal(0, 1, n)                           # 참 효과 -1
    return Z, t, y


def test_naive_by_hand():
    y = np.array([1.0, 2.0, 3.0, 6.0])
    t = np.array([0, 0, 1, 1])
    assert np.isclose(mod.naive_difference(y, t), 4.5 - 1.5)


def test_ipw_by_hand():
    y = np.array([1.0, 3.0, 2.0, 4.0])
    t = np.array([1, 1, 0, 0])
    e = np.array([0.5, 0.25, 0.5, 0.75])
    # 처치군: 가중치 2, 4 → (2·1 + 4·3)/6 = 7/3. 대조군: 가중치 2, 4 → (2·2 + 4·4)/6 = 10/3
    assert np.isclose(mod.ipw_effect(y, t, e), 7 / 3 - 10 / 3), "가중치는 처치군 1/e, 대조군 1/(1-e)이고 가중 평균은 가중치 합으로 나눕니다"


def test_propensity_score_equations():
    Z, t, _ = confounded(0, n=3000)
    e = mod.propensity(Z[:, None], t)
    assert np.shape(e) == (3000,)
    X = np.c_[np.ones(3000), Z]
    assert np.allclose(X.T @ (t - e), 0, atol=1e-6), "최대우도에서 X^T (t - e) = 0이어야 합니다(절편 포함 확인)"
    assert np.all((e > 0) & (e < 1))


def test_ipw_removes_confounding():
    Z, t, y = confounded(1)
    naive = mod.naive_difference(y, t)
    est = mod.ipw_effect(y, t, mod.propensity(Z[:, None], t))
    assert naive > 0, "이 데이터에서 단순 비교는 부호까지 틀려야 합니다(교란)"
    assert abs(est - (-1.0)) < 0.15, f"IPW 추정은 참 효과 -1 근처여야 합니다. 현재 {est:.3f}"
