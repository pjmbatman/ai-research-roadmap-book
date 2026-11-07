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


satisfies = load().satisfies

# 기대값은 packaging 라이브러리(SpecifierSet)의 판정과 같다.
CASES = [
    ("1.26.4", "==1.26.4", True),
    ("1.26.4", "!=1.26.4", False),
    ("2.5", ">=2.0,<3", True),
    ("3.0", ">=2.0,<3", False),
    ("2.5.9", "<=2.5.9", True),
    ("1.0.1", ">1.0", True),
    ("2.6.0", ">=1.23, <2.6", False),
]


@pytest.mark.parametrize("version,spec,expected", CASES)
def test_basic_operators(version, spec, expected):
    assert satisfies(version, spec) is expected, f"{version} {spec} -> {expected}"


def test_numbers_not_strings():
    assert satisfies("1.10", ">1.9") is True, "1.10은 1.9보다 큽니다. 문자열이 아니라 정수로 비교하세요"
    assert satisfies("1.9", ">=1.10") is False
    assert satisfies("12.10", ">12.4") is True


def test_zero_padding():
    assert satisfies("2.0.0", "==2.0") is True, "2.0과 2.0.0은 같은 버전입니다(뒤에 0을 채워 비교)"
    assert satisfies("2.0.1", "==2.0") is False
    assert satisfies("2.0", "!=2.0.0") is False
    assert satisfies("1.0", "<1.0.0") is False


def test_wildcard():
    assert satisfies("1.26.4", "==1.26.*") is True
    assert satisfies("1.27.0", "==1.26.*") is False
    assert satisfies("1.2.6", "==1.26.*") is False, "1.2.6은 1.26.*이 아닙니다. 자리 단위로 비교하세요"
    assert satisfies("1.26.1", "!=1.26.*") is False
    assert satisfies("2.0", "!=1.26.*") is True


def test_compatible_release():
    assert satisfies("1.4.2", "~=1.4.2") is True
    assert satisfies("1.4.10", "~=1.4.2") is True
    assert satisfies("1.5.0", "~=1.4.2") is False, "~=1.4.2는 1.4.* 안에서만 올라갑니다"
    assert satisfies("1.4.1", "~=1.4.2") is False
    assert satisfies("2.9", "~=2.2") is True, "~=2.2는 2.* 안에서 2.2 이상입니다"
    assert satisfies("3.0", "~=2.2") is False
    assert satisfies("2.1", "~=2.2") is False
