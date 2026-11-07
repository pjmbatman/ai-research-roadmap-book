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


gaussian_kl = load().gaussian_kl


def numeric_kl(m1, s1, m2, s2):
    x = np.linspace(min(m1, m2) - 12 * max(s1, s2), max(m1, m2) + 12 * max(s1, s2), 200001)
    logp = -0.5 * ((x - m1) / s1) ** 2 - np.log(s1 * np.sqrt(2 * np.pi))
    logq = -0.5 * ((x - m2) / s2) ** 2 - np.log(s2 * np.sqrt(2 * np.pi))
    return np.sum(np.exp(logp) * (logp - logq)) * (x[1] - x[0])


@pytest.mark.parametrize("m1,s1,m2,s2", [(0, 1, 0, 1), (0, 1, 1, 1), (0.5, 0.3, -1, 2.0), (2, 1.5, 2, 0.7)])
def test_matches_numeric(m1, s1, m2, s2):
    assert np.isclose(float(gaussian_kl(m1, s1, m2, s2)), numeric_kl(m1, s1, m2, s2), atol=1e-6), \
        "수치적분 값과 다릅니다. 식의 각 항(로그 비, 분산, 평균 차이)을 확인하세요"


def test_zero_for_same():
    assert np.isclose(float(gaussian_kl(1.3, 0.4, 1.3, 0.4)), 0.0), "같은 분포의 KL은 0이어야 합니다"


def test_asymmetry():
    narrow_to_wide = float(gaussian_kl(0, 0.1, 0, 1.0))
    wide_to_narrow = float(gaussian_kl(0, 1.0, 0, 0.1))
    assert not np.isclose(narrow_to_wide, wide_to_narrow), "KL은 대칭이 아닙니다"
    assert wide_to_narrow > narrow_to_wide, "넓은 p를 좁은 q로 재면(q가 p를 덮지 못하면) 더 크게 벌해야 합니다"


def test_unit_invariance():
    a = float(gaussian_kl(0.3, 0.2, -0.1, 0.5))
    b = float(gaussian_kl(300.0, 200.0, -100.0, 500.0))   # 같은 분포를 1000배 단위로
    assert np.isclose(a, b), "평균·표준편차를 같은 배율로 바꾸면 KL은 변하지 않아야 합니다"


def test_vectorized():
    out = gaussian_kl(np.zeros(3), np.ones(3), np.array([0.0, 1.0, 2.0]), np.ones(3))
    assert np.shape(out) == (3,), "배열 인자는 원소별로 계산해야 합니다"
    assert np.allclose(out, [0.0, 0.5, 2.0])
