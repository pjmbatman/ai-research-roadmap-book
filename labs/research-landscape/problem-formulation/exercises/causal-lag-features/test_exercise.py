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


lag_features = load().lag_features


def test_values_and_shape():
    x = np.arange(6.0)                                     # [0, 1, 2, 3, 4, 5]
    F = lag_features(x, [0, 1, 3])
    assert np.shape(F) == (6, 3), f"shape은 (T, len(lags)) = (6, 3)이어야 합니다. 현재 {np.shape(F)}"
    assert np.allclose(F[:, 0], x), "lag 0은 현재 값입니다"
    assert np.allclose(F[1:, 1], x[:-1]), "F[t, j] = x[t - lags[j]]입니다"
    assert np.allclose(F[3:, 2], x[:3]), "lag 3은 3단계 전의 값입니다"


def test_missing_values_are_nan():
    F = lag_features(np.arange(5.0), [2])
    assert np.all(np.isnan(F[:2, 0])), "과거 값이 없는 앞쪽 칸은 NaN이어야 합니다(0이나 당겨 채운 값 금지)"
    assert not np.any(np.isnan(F[2:, 0])), "값이 있는 칸에 NaN이 있으면 안 됩니다"


def test_lag_longer_than_series():
    F = lag_features(np.arange(3.0), [5])
    assert np.shape(F) == (3, 1) and np.all(np.isnan(F)), "시계열보다 긴 lag은 모두 NaN입니다"


@pytest.mark.parametrize("t", [0, 7, 30, 48])
def test_future_values_do_not_change_past_rows(t):
    rng = np.random.default_rng(t)
    x = rng.normal(size=50)                                # (50,)
    F = lag_features(x, [0, 1, 2, 5])
    x_changed = x.copy()
    x_changed[t + 1:] = rng.normal(size=50 - t - 1) * 100  # 시각 t 이후만 바꾼다
    G = lag_features(x_changed, [0, 1, 2, 5])
    assert np.allclose(F[: t + 1], G[: t + 1], equal_nan=True), f"시각 {t} 이후의 값이 F[:{t + 1}]에 영향을 주면 안 됩니다"


def test_negative_lag_rejected():
    with pytest.raises(ValueError):
        lag_features(np.arange(10.0), [0, -1])
