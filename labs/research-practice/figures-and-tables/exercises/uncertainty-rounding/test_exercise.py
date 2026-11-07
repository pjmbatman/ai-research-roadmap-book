import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


format_pm = load().format_pm


@pytest.mark.parametrize("mean,err,sig,expected", [
    (0.41325281, 0.04468845, 2, "0.413 ± 0.045"),
    (0.39167009, 0.04899274, 2, "0.392 ± 0.049"),
    (9.3617, 0.79994, 2, "9.36 ± 0.80"),
    (0.123456, 0.0123, 1, "0.12 ± 0.01"),
])
def test_basic(mean, err, sig, expected):
    assert format_pm(mean, err, sig) == expected


def test_default_sig_is_two():
    assert format_pm(0.41325281, 0.04468845) == "0.413 ± 0.045"


def test_keeps_trailing_zero():
    assert format_pm(11.68, 1.0021, 2) == "11.7 ± 1.0", "불확실성의 끝자리 0도 유효숫자이므로 지우지 않습니다"


def test_rounds_left_of_decimal_point():
    assert format_pm(1234.0, 156.0, 2) == "1230 ± 160", "자릿수가 음수이면 십의 자리에서 반올림합니다"
    assert format_pm(98765.0, 4321.0, 1) == "99000 ± 4000"


def test_rounding_carries_to_next_decade():
    assert format_pm(0.52, 0.0996, 1) == "0.5 ± 0.1", "0.0996 → 0.1이 되면 자릿수를 다시 정해야 합니다(0.10은 유효숫자 2자리)"
    assert format_pm(3.14159, 0.0997, 2) == "3.14 ± 0.10"


def test_no_negative_zero():
    assert format_pm(-0.0004, 0.012, 2) == "0.000 ± 0.012", "-0.000이 아니라 0.000으로 씁니다"


def test_negative_mean():
    assert format_pm(-0.0386, 0.0167, 2) == "-0.039 ± 0.017"


@pytest.mark.parametrize("err", [0.0, -0.1])
def test_nonpositive_error_raises(err):
    with pytest.raises(ValueError):
        format_pm(1.0, err)
