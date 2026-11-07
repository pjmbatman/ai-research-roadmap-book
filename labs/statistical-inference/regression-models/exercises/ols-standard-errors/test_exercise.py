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


m = load()


def test_hand_example():
    # x = 0, 1, 2, 3, y = 1, 3, 2, 4: 기울기 0.8, 절편 1.3, RSS = 1.8, σ̂² = 1.8 / 2 = 0.9
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    r = m.ols_fit(X, [1.0, 3.0, 2.0, 4.0])
    assert np.allclose(r["beta"], [1.3, 0.8])
    assert r["sigma2"] == pytest.approx(0.9), "잔차 분산은 n - p로 나눕니다"
    assert r["se"][1] == pytest.approx(np.sqrt(0.9 / 5)), "기울기의 SE는 sqrt(σ̂² / Σ(x - x̄)²)입니다"


def test_matches_lstsq():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(0, 0.3, 50)
    r = m.ols_fit(X, y)
    assert np.allclose(r["beta"], np.linalg.lstsq(X, y, rcond=None)[0])
    assert r["se"].shape == (4,)


def test_se_matches_sampling_spread():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 25)
    X = np.column_stack([np.ones_like(x), x])
    fits = [m.ols_fit(X, 1.0 + 0.5 * x + rng.normal(0, 2.0, x.size)) for _ in range(3000)]
    slopes = np.array([f["beta"][1] for f in fits])
    ses = np.array([f["se"][1] for f in fits])
    assert ses.mean() == pytest.approx(slopes.std(), rel=0.06), "보고된 SE는 반복 실험에서 기울기의 실제 표준편차와 맞아야 합니다"
    assert np.mean([f["sigma2"] for f in fits]) == pytest.approx(4.0, rel=0.03), "n - p로 나눈 σ̂²는 불편 추정량입니다"
