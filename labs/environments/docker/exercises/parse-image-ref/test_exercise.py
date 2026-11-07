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


parse = load().parse
D = "sha256:" + "ab" * 32


@pytest.mark.parametrize("ref,expected", [
    ("ubuntu", ("docker.io", "library/ubuntu", "latest", None)),
    ("python:3.12-slim", ("docker.io", "library/python", "3.12-slim", None)),
    ("ghcr.io/astral-sh/uv:latest", ("ghcr.io", "astral-sh/uv", "latest", None)),
    ("myuser/proj", ("docker.io", "myuser/proj", "latest", None)),
    ("registry.example.org/lab/team/proj:v1.2", ("registry.example.org", "lab/team/proj", "v1.2", None)),
])
def test_tag_and_defaults(ref, expected):
    r = parse(ref)
    assert (r["registry"], r["repository"], r["tag"], r["digest"]) == expected, f"{ref} -> {expected}"


def test_registry_port_is_not_a_tag():
    r = parse("localhost:5000/proj")
    assert r["registry"] == "localhost:5000", "첫 부분의 ':5000'은 registry의 포트입니다"
    assert r["repository"] == "proj"
    assert r["tag"] == "latest", "마지막 '/' 뒤에 ':'가 없으면 tag가 없는 것입니다"
    r = parse("localhost:5000/proj:dev")
    assert (r["registry"], r["repository"], r["tag"]) == ("localhost:5000", "proj", "dev")


def test_digest_only():
    r = parse(f"python@{D}")
    assert r["digest"] == D
    assert r["tag"] is None, "digest만 있으면 tag는 None입니다(latest를 붙이지 않습니다)"
    assert r["repository"] == "library/python"


def test_tag_and_digest():
    r = parse(f"ghcr.io/astral-sh/uv:0.9@{D}")
    assert (r["registry"], r["repository"], r["tag"], r["digest"]) == ("ghcr.io", "astral-sh/uv", "0.9", D)


def test_localhost_without_port():
    r = parse("localhost/proj:dev")
    assert (r["registry"], r["repository"], r["tag"]) == ("localhost", "proj", "dev")
