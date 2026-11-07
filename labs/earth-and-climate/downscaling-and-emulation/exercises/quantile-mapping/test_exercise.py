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


quantile_map = load().quantile_map

RNG = np.random.default_rng(0)
OBS = RNG.gamma(2.0, 3.0, 20000)            # 관측: 오른쪽 꼬리가 긴 분포
MOD = 0.6 * RNG.gamma(3.0, 2.0, 20000) + 1.0   # 모델: 평균·퍼짐·모양이 모두 다르다


def test_matches_observed_quantiles():
    out = quantile_map(MOD, MOD, OBS)
    ps = [5, 25, 50, 75, 95, 99]
    got, want = np.percentile(out, ps), np.percentile(OBS, ps)
    assert np.allclose(got, want, rtol=0.02, atol=0.05), f"보정한 모델의 분위수 {np.round(got, 2)}가 관측 {np.round(want, 2)}와 같아야 합니다"


def test_shape_and_monotone():
    x = np.linspace(MOD.min(), MOD.max(), 50).reshape(5, 10)
    out = quantile_map(x, MOD, OBS)
    assert np.shape(out) == (5, 10), "반환 shape은 x와 같아야 합니다"
    assert np.all(np.diff(out.ravel()) >= -1e-12), "quantile mapping은 순서를 보존해야 합니다(단조 증가)"


def test_pure_shift_is_removed():
    out = quantile_map(OBS[:5000] + 2.0, OBS + 2.0, OBS)
    assert np.allclose(out, OBS[:5000], atol=0.1), "모델이 관측보다 일정하게 2만큼 크면, 보정은 2를 빼는 것과 같아야 합니다"


def test_identity_when_same_distribution():
    x = np.array([1.0, 5.0, 10.0])
    assert np.allclose(quantile_map(x, OBS, OBS), x, atol=0.05)


def test_out_of_range_is_clamped():
    big = MOD.max() + 10.0
    out = quantile_map(np.array([big, MOD.min() - 1.0]), MOD, OBS)
    assert np.isclose(out[0], OBS.max()), "보정 기간 모델 최댓값보다 큰 값은 관측 최댓값으로 고정됩니다(np.interp의 성질)"
    assert np.isclose(out[1], OBS.min())
