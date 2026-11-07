import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from halflife import fit_half_life, simulate  # noqa: E402


def test_noise_free_exact():
    t_s = np.linspace(0, 300, 31)
    counts = 1e4 * np.exp(-np.log(2) / 60.0 * t_s)
    assert fit_half_life(t_s, counts) == pytest.approx(60.0, rel=1e-9)


@pytest.mark.parametrize("half_life_s", [30.0, 120.0, 400.0])
def test_poisson_within_5_percent(half_life_s):
    t_s, counts = simulate(half_life_s, seed=1)
    assert fit_half_life(t_s, counts) == pytest.approx(half_life_s, rel=0.05)


def test_zero_counts_are_skipped():
    t_s, counts = simulate(20.0, n0=50, seed=2)      # 뒤쪽은 0 계수가 많다
    assert (counts == 0).any()
    assert np.isfinite(fit_half_life(t_s, counts))
