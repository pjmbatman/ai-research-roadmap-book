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
rng = np.random.default_rng(0)
x = np.r_[np.linspace(0, 10, 29), 25.0]                  # 마지막 점만 멀리 떨어진 입력
X = np.column_stack([np.ones_like(x), x])
y = 1.0 + 0.5 * x + rng.normal(0, 0.5, x.size)


def test_leverage_properties():
    h = m.leverage(X)
    assert h.shape == (30,)
    assert h.sum() == pytest.approx(2.0), "leverage의 합은 계수의 수 p입니다"
    assert np.all((h > 0) & (h < 1))
    assert np.argmax(h) == 29, "입력이 멀리 떨어진 점의 leverage가 가장 큽니다"


def test_leverage_simple_regression_formula():
    h = m.leverage(X)
    expected = 1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    assert np.allclose(h, expected)


def test_cooks_distance_matches_refitting():
    y_bad = y.copy()
    y_bad[29] -= 6.0                                      # leverage가 큰 점에 오류
    D = m.cooks_distance(X, y_bad)
    beta = np.linalg.lstsq(X, y_bad, rcond=None)[0]
    s2 = np.sum((y_bad - X @ beta) ** 2) / (30 - 2)
    for i in (0, 15, 29):
        keep = np.arange(30) != i
        b_i = np.linalg.lstsq(X[keep], y_bad[keep], rcond=None)[0]
        assert D[i] == pytest.approx(np.sum((X @ beta - X @ b_i) ** 2) / (2 * s2)), "닫힌 꼴은 다시 적합한 결과와 같아야 합니다"
    assert np.argmax(D) == 29 and D[29] > 1


def test_same_residual_matters_more_at_high_leverage():
    y1, y2 = y.copy(), y.copy()
    y1[14] += 3.0                                          # 가운데 점의 오류
    y2[29] += 3.0                                          # 가장자리 점의 같은 크기 오류
    assert m.cooks_distance(X, y2)[29] > 3 * m.cooks_distance(X, y1)[14]
