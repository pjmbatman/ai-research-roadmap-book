import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


find_shadowed = load().find_shadowed

USER = "/home/me/.local/lib/python3.12/site-packages"
ENV = "/home/me/proj/.venv/lib/python3.12/site-packages"
SYS = "/usr/lib/python3/dist-packages"


def test_user_site_shadows_system():
    r = find_shadowed([USER, SYS], {USER: {"numpy": "1.26.4"}, SYS: {"numpy": "2.1.0", "scipy": "1.13.0"}})
    assert set(r) == {"numpy"}, "한 곳에만 있는 scipy는 가려진 패키지가 아닙니다"
    assert r["numpy"]["used"] == ("1.26.4", USER), "sys_path에서 앞선 디렉터리의 것이 import됩니다"
    assert r["numpy"]["shadowed"] == [("2.1.0", SYS)]


def test_order_matters():
    installed = {USER: {"numpy": "1.26.4"}, SYS: {"numpy": "2.1.0"}}
    assert find_shadowed([SYS, USER], installed)["numpy"]["used"] == ("2.1.0", SYS)


def test_three_places_in_search_order():
    installed = {ENV: {"torch": "2.4.0"}, USER: {"torch": "2.1.0"}, SYS: {"torch": "1.13.1"}}
    r = find_shadowed([ENV, USER, SYS], installed)
    assert r["torch"]["used"] == ("2.4.0", ENV)
    assert r["torch"]["shadowed"] == [("2.1.0", USER), ("1.13.1", SYS)], "가려진 것들도 검색 순서대로 담습니다"


def test_dirs_not_on_path_are_ignored():
    installed = {ENV: {"pandas": "2.2.2"}, USER: {"pandas": "1.5.3"}}
    assert find_shadowed([ENV, SYS], installed) == {}, "sys_path에 없는 디렉터리는 import 대상이 아닙니다"


def test_name_normalization():
    installed = {ENV: {"scikit-learn": "1.5.0"}, USER: {"Scikit_Learn": "1.3.2"}}
    r = find_shadowed([ENV, USER], installed)
    assert list(r) == ["scikit-learn"], "대소문자와 -/_를 구분하지 않고, 키는 소문자·하이픈으로 정규화합니다"
    assert r["scikit-learn"]["shadowed"] == [("1.3.2", USER)]


def test_no_duplicates():
    assert find_shadowed([ENV], {ENV: {"numpy": "2.0.0"}}) == {}
