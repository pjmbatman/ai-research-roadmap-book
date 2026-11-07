import importlib.util
import itertools
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


m = load()


def design(k, reps=1):
    return np.repeat(np.array(list(itertools.product([0, 1], repeat=k))), reps, axis=0)


def test_main_effects_additive():
    L = design(3)
    y = 1.0 + 0.5 * L[:, 0] - 0.2 * L[:, 1] + 0.0 * L[:, 2]
    e = m.factorial_effects(L, y)
    assert e[(0,)] == pytest.approx(0.5) and e[(1,)] == pytest.approx(-0.2) and e[(2,)] == pytest.approx(0.0)
    assert e[(0, 1)] == pytest.approx(0.0) and e[(0, 2)] == pytest.approx(0.0) and e[(1, 2)] == pytest.approx(0.0)


def test_interaction():
    L = design(2)
    y = 0.002 * L[:, 0] + 0.020 * L[:, 1] + 0.010 * L[:, 0] * L[:, 1]
    e = m.factorial_effects(L, y)
    assert e[(0,)] == pytest.approx(0.002 + 0.010 / 2), "주효과는 다른 요인의 두 수준에 대한 평균입니다"
    assert e[(1,)] == pytest.approx(0.020 + 0.010 / 2)
    assert e[(0, 1)] == pytest.approx(0.010 / 2), "교호작용은 (j의 효과 | i=1) - (j의 효과 | i=0)의 절반입니다"


def test_keys():
    e = m.factorial_effects(design(4), np.arange(16.0))
    assert set(e) == {(0,), (1,), (2,), (3,)} | set(itertools.combinations(range(4), 2))


def test_replicates_and_noise():
    rng = np.random.default_rng(0)
    L = design(3, reps=200)
    y = 0.8 + 0.03 * L[:, 0] + 0.01 * L[:, 2] + 0.02 * L[:, 0] * L[:, 2] + rng.normal(0, 0.005, len(L))
    e = m.factorial_effects(L, y)
    assert e[(0,)] == pytest.approx(0.04, abs=1e-3)
    assert e[(2,)] == pytest.approx(0.02, abs=1e-3)
    assert e[(0, 2)] == pytest.approx(0.01, abs=1e-3)
    assert e[(1,)] == pytest.approx(0.0, abs=1e-3)


def test_ofat_misses_interaction():
    # 하나씩 바꾸기: (0,0)에서 출발해 요인 0만 바꾼 차이는 주효과와 다르다.
    L = design(2)
    y = 0.002 * L[:, 0] + 0.020 * L[:, 1] + 0.010 * L[:, 0] * L[:, 1]
    ofat = y[2] - y[0]                                    # (1,0) - (0,0)
    e = m.factorial_effects(L, y)
    assert ofat == pytest.approx(0.002)
    assert e[(0,)] > 3 * ofat
