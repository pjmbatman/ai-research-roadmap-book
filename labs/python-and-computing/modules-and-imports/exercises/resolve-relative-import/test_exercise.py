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


resolve = load().resolve


@pytest.mark.parametrize(
    "name,package,expected",
    [
        (".physics", "heatlab.models", "heatlab.models.physics"),
        ("..utils", "heatlab.models", "heatlab.utils"),
        ("..io.hdf5", "heatlab.models", "heatlab.io.hdf5"),
        (".", "heatlab.models", "heatlab.models"),
        ("..", "heatlab.models", "heatlab"),
        ("numpy", "heatlab.models", "numpy"),
    ],
)
def test_table(name, package, expected):
    assert resolve(name, package) == expected


@pytest.mark.parametrize(
    "name,package",
    [
        (".a", "p"),
        ("..a", "p.q"),
        ("...a.b", "p.q.r"),
        (".a.b.c", "p.q"),
        ("....", "p.q.r.s"),
        ("scipy.sparse", "p.q"),
    ],
)
def test_matches_importlib(name, package):
    assert resolve(name, package) == importlib.util.resolve_name(name, package)


def test_package_is_not_module_name():
    # 모듈 heatlab/models/cnn.py의 __package__는 "heatlab.models"다
    assert resolve("..utils", "heatlab.models") == "heatlab.utils", "점 둘은 패키지에서 한 단계 위입니다"


@pytest.mark.parametrize("package", [None, ""])
def test_no_parent_package(package):
    with pytest.raises(ImportError):
        resolve(".physics", package)


@pytest.mark.parametrize("name,package", [("...a", "p.q"), ("..", "p")])
def test_beyond_top_level(name, package):
    with pytest.raises(ImportError):
        resolve(name, package)


def test_absolute_name_needs_no_package():
    assert resolve("json", None) == "json"
