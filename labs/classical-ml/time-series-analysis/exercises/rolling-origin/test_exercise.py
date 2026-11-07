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


def persistence(h):
    return lambda hist: np.full(h, hist[-1])


def test_origins_and_history_lengths():
    seen = []

    def spy(hist):
        seen.append(len(hist))
        return np.zeros(2)

    E = mod.rolling_origin_errors(np.arange(10.0), spy, initial=4, horizon=2, step=1)
    # 원점 o = 3, 4, 5, 6, 7 (o + 2 <= 9), 예측기가 받는 길이는 o + 1
    assert seen == [4, 5, 6, 7, 8], f"예측기가 받은 데이터 길이가 [4, 5, 6, 7, 8]이어야 합니다. 현재 {seen}"
    assert np.shape(E) == (5, 2)


def test_step():
    seen = []

    def spy(hist):
        seen.append(len(hist))
        return np.zeros(3)

    mod.rolling_origin_errors(np.arange(20.0), spy, initial=5, horizon=3, step=4)
    assert seen == [5, 9, 13, 17], f"step=4이면 원점이 4씩 늘어야 합니다. 현재 {seen}"


def test_persistence_errors_by_hand():
    y = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    E = mod.rolling_origin_errors(y, persistence(2), initial=2, horizon=2)
    # o = 1: 예측 [1, 1], 실제 [3, 6] → [-2, -5]; o = 2: 예측 [3, 3], 실제 [6, 10] → [-3, -7]
    assert np.allclose(E, [[-2, -5], [-3, -7]]), f"오차는 예측 - 실제입니다. 현재 {np.asarray(E).tolist()}"
    assert np.allclose(mod.horizon_rmse(E), [np.sqrt((4 + 9) / 2), np.sqrt((25 + 49) / 2)])


def test_no_future_given_to_forecaster():
    y = np.arange(50.0)

    def cheat_check(hist):
        assert hist[-1] == len(hist) - 1, "예측기에는 원점까지의 값만 넘겨야 합니다"
        return np.full(5, hist[-1])

    mod.rolling_origin_errors(y, cheat_check, initial=10, horizon=5, step=3)


def test_random_walk_rmse_grows_like_sqrt_h():
    y = np.cumsum(np.random.default_rng(0).normal(size=4000))
    r = mod.horizon_rmse(mod.rolling_origin_errors(y, persistence(16), initial=100, horizon=16, step=2))
    assert np.shape(r) == (16,)
    assert abs(r[15] / r[3] - 2.0) < 0.25, f"random walk에서 persistence의 RMSE는 sqrt(h)에 비례합니다(h=16/h=4 비율 약 2). 현재 {r[15] / r[3]:.2f}"
