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


lookup = load().lookup

FILES = {
    "/usr/local/cuda-12.1/bin/nvcc",
    "/usr/local/cuda-12.4/bin/nvcc",
    "/usr/local/cuda-12.1/lib64/libcudart.so.12",
    "/usr/local/cuda-12.4/lib64/libcudart.so.12",
    "/usr/bin/python3",
}
LINKS = {"/usr/local/cuda": "/usr/local/cuda-12.4"}


def test_symlink_is_followed():
    assert lookup("nvcc", "/usr/local/cuda/bin:/usr/bin", FILES, LINKS) == (
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
    ), "/usr/local/cuda는 cuda-12.4를 가리키는 링크입니다"


def test_first_match_wins():
    r = lookup("nvcc", "/usr/local/cuda-12.1/bin:/usr/local/cuda/bin", FILES, LINKS)
    assert r == ("/usr/local/cuda-12.1/bin/nvcc", "/usr/local/cuda-12.1/bin/nvcc"), "PATH의 앞쪽 항목이 이깁니다"
    r = lookup("nvcc", "/usr/local/cuda/bin:/usr/local/cuda-12.1/bin", FILES, LINKS)
    assert r[1] == "/usr/local/cuda-12.4/bin/nvcc", "순서를 바꾸면 결과도 바뀝니다"


def test_library_path_same_rule():
    r = lookup("libcudart.so.12", "/usr/local/cuda-12.1/lib64:/usr/local/cuda/lib64", FILES, LINKS)
    assert r[1] == "/usr/local/cuda-12.1/lib64/libcudart.so.12"


def test_not_found():
    assert lookup("nvcc", "/usr/bin:/opt/bin", FILES, LINKS) is None


def test_empty_entries_and_trailing_slash():
    r = lookup("python3", "::/usr/bin/:", FILES, LINKS)
    assert r == ("/usr/bin/python3", "/usr/bin/python3"), "빈 항목은 건너뛰고, 끝의 /는 무시합니다"


def test_link_matches_whole_directory_only():
    links = {"/usr/local/cuda": "/usr/local/cuda-12.4", "/opt/cuda-12": "/usr/local/cuda-12.4"}
    r = lookup("nvcc", "/usr/local/cuda-12.1/bin", FILES, links)
    assert r == ("/usr/local/cuda-12.1/bin/nvcc", "/usr/local/cuda-12.1/bin/nvcc"), (
        "/usr/local/cuda 링크가 /usr/local/cuda-12.1에 적용되면 안 됩니다"
    )
    assert lookup("nvcc", "/opt/cuda-12/bin", FILES, links)[1] == "/usr/local/cuda-12.4/bin/nvcc"
