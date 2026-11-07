import importlib.util
import os
from decimal import Decimal, getcontext
from pathlib import Path

import pytest

HERE = Path(__file__).parent
getcontext().prec = 60


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


quadratic_roots = load().quadratic_roots


def exact_roots(a, b, c):
    """60자리 십진 연산으로 계산한 참값(작은 근, 큰 근)."""
    a, b, c = Decimal(a), Decimal(b), Decimal(c)
    d = (b * b - 4 * a * c).sqrt()
    r = sorted([(-b - d) / (2 * a), (-b + d) / (2 * a)])
    return float(r[0]), float(r[1])


CASES = [
    (1.0, -3.0, 2.0),          # 근 1, 2
    (1.0, -1e8, 1.0),          # 작은 근 약 1e-8: 근의 공식이면 상대 오차 25%
    (1.0, 1e8, 1.0),           # b의 부호가 반대
    (2.0, 0.0, -8.0),          # b = 0, 근 -2, 2
    (1e-3, 1e5, -2.5),         # a가 작음
    (-3.0, 7e6, 4e-2),         # a가 음수
]


@pytest.mark.parametrize("a,b,c", CASES)
def test_relative_accuracy(a, b, c):
    got = quadratic_roots(a, b, c)
    want = exact_roots(a, b, c)
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-14 * abs(w), f"근 {g:.17g}의 상대 오차가 큽니다 (참값 {w:.17g})"


def test_sorted_order():
    x1, x2 = quadratic_roots(1.0, -3.0, 2.0)
    assert x1 < x2, "작은 근을 먼저 반환해야 합니다"


def test_small_root_not_cancelled():
    x1, _ = quadratic_roots(1.0, -1e8, 1.0)
    assert abs(x1 - 1e-8) / 1e-8 < 1e-14, f"작은 근이 {x1:.6e}입니다. 근의 공식의 상쇄를 피했는지 확인하세요"
