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


mode_to_octal = load().mode_to_octal


@pytest.mark.parametrize("s,expected", [
    ("rwxr-xr-x", "755"),
    ("rw-r--r--", "644"),
    ("rw-------", "600"),
    ("rw-r-----", "640"),
    ("---------", "000"),
])
def test_basic(s, expected):
    assert mode_to_octal(s) == expected, f"{s}는 {expected}입니다. 현재 {mode_to_octal(s)}"


def test_leading_type_char():
    assert mode_to_octal("drwxr-x---") == "750", "맨 앞의 파일 종류 글자(d)는 권한이 아닙니다"
    assert mode_to_octal("-rw-r--r--") == "644"
    assert mode_to_octal("lrwxrwxrwx") == "777"


def test_setgid_shared_directory():
    assert mode_to_octal("drwxrwsr-x") == "2775", "그룹 실행 자리의 소문자 s = setgid(2) + 실행(1)"


def test_sticky_tmp():
    assert mode_to_octal("drwxrwxrwt") == "1777", "기타 실행 자리의 소문자 t = sticky(1) + 실행(1)"


def test_setuid():
    assert mode_to_octal("-rwsr-xr-x") == "4755", "소유자 실행 자리의 소문자 s = setuid(4) + 실행(1)"


def test_uppercase_means_no_execute():
    assert mode_to_octal("drwxrwSr-x") == "2765", "대문자 S는 setgid만 있고 그룹 실행 비트는 꺼져 있습니다"
    assert mode_to_octal("drwxrwxrwT") == "1776", "대문자 T는 sticky만 있고 기타 실행 비트는 꺼져 있습니다"


@pytest.mark.parametrize("bad", ["rwx", "rwxr-xr-x--", ""])
def test_invalid_length(bad):
    with pytest.raises(ValueError):
        mode_to_octal(bad)
