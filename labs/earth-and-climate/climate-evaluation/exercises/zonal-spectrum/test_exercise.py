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


energy_ratio = load().energy_ratio

NLON = 64
LON = np.arange(NLON) * 2 * np.pi / NLON


def random_field(n, seed):
    """k^-1.5 진폭, 무작위 위상의 주기 장 (n, NLON)."""
    g = np.random.default_rng(seed)
    k = np.arange(1, NLON // 2 + 1)
    ph = g.uniform(0, 2 * np.pi, (n, k.size))
    return (k**-1.5 * np.cos(k * LON[:, None, None] + ph)).sum(-1).T


def test_identical_is_one():
    f = random_field(20, 0)
    ratio, k_half = energy_ratio(f, f)
    assert np.shape(ratio) == (NLON // 2,), f"ratio의 길이는 nlon // 2 = {NLON // 2}이어야 합니다"
    assert np.allclose(ratio, 1) and k_half is None


def test_single_wave():
    k = np.arange(1, NLON // 2 + 1)
    amp = k**-1.5
    truth = (amp * np.cos(k * LON[:, None] + 0.3)).sum(-1)[None]          # (1, NLON) 모든 파수에 에너지
    amp_fc = amp.copy()
    amp_fc[2] *= 0.5                                                       # k = 3 성분만 진폭 절반
    fc = (amp_fc * np.cos(k * LON[:, None] + 0.3)).sum(-1)[None]
    ratio, k_half = energy_ratio(fc, truth)
    assert np.allclose(np.delete(ratio, 2), 1), "k = 3 밖의 파수는 그대로여야 합니다"
    assert np.isclose(ratio[2], 0.25), "진폭을 절반으로 줄이면 에너지는 4분의 1이 됩니다(k = 3은 ratio[2])"
    assert k_half == 3


def test_smoothing_loses_small_scales():
    f = random_field(50, 1)
    kern = np.array([1, 2, 3, 2, 1]) / 9                    # 경도 방향 이동평균(주기 경계)
    smooth = sum(w * np.roll(f, s, axis=-1) for w, s in zip(kern, range(-2, 3)))
    ratio, k_half = energy_ratio(smooth, f)
    assert ratio[0] > 0.9 and ratio[-1] < 0.05, "평활은 큰 규모를 남기고 작은 규모의 에너지를 지웁니다"
    assert k_half is not None and 5 <= k_half <= 15


def test_leading_axes_averaged():
    f = random_field(24, 2).reshape(2, 3, 4, NLON)
    g = 0.8 * f
    ratio, _ = energy_ratio(g, f)
    assert np.allclose(ratio, 0.64), "앞쪽 축 전체에 대해 평균한 에너지를 비교해야 합니다"
