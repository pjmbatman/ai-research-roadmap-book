import importlib.util
import math
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


cohen_kappa = load().cohen_kappa


def test_perfect_agreement():
    a = [0, 1, 1, 0, 2, 2, 1]
    assert cohen_kappa(a, a) == pytest.approx(1.0), "판정이 모두 같으면 kappa는 1이어야 합니다"


def test_chapter_example():
    r1 = [1, 1, 1, 1] + [0] * 16
    r2 = [1, 1, 0, 0, 1, 1] + [0] * 14
    assert cohen_kappa(r1, r2) == pytest.approx(0.375), "본문 예: p_o = 0.80, p_e = 0.68이면 kappa = 0.375입니다"


def test_constant_rater_has_zero_kappa():
    r1 = [1, 1, 1, 1] + [0] * 16
    r3 = [0] * 20
    k = cohen_kappa(r1, r3)
    assert abs(k) < 1e-12, f"무조건 거절하는 심사자와의 kappa는 0이어야 합니다(일치율은 0.80이지만 모두 우연). 현재 {k}"


def test_three_string_categories():
    a = ["acc", "acc", "weak", "weak", "rej", "rej", "rej", "acc"]
    b = ["acc", "weak", "weak", "weak", "rej", "acc", "rej", "acc"]
    # p_o = 6/8, p_e = (3*3 + 2*3 + 3*2) / 64 = 21/64  ->  kappa = 27/43
    assert cohen_kappa(a, b) == pytest.approx(27 / 43), "범주가 셋 이상이고 문자열이어도 같은 식으로 계산합니다"


def test_category_used_by_only_one_rater():
    a = ["acc", "rej", "rej", "rej"]
    b = ["weak", "rej", "rej", "acc"]
    # p_o = 2/4, p_e = (1/4)(1/4) + (3/4)(2/4) + 0 = 7/16  ->  kappa = (8/16 - 7/16) / (9/16) = 1/9
    assert cohen_kappa(a, b) == pytest.approx(1 / 9), "한 심사자만 쓴 범주도 p_e의 합에 넣어야 합니다(그 항은 0)"


def test_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, size=50), rng.integers(0, 3, size=50)
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a)), "kappa는 두 심사자의 순서와 무관해야 합니다"


def test_undefined_when_both_use_one_category():
    k = cohen_kappa([0, 0, 0], [0, 0, 0])
    assert isinstance(k, float) and math.isnan(k), "두 심사자가 같은 한 범주만 쓰면 p_e = 1이므로 nan을 반환합니다"


def test_length_mismatch():
    with pytest.raises(ValueError):
        cohen_kappa([0, 1, 1], [0, 1])
