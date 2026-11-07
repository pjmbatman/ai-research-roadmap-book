import hashlib
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


dataset_manifest = load().dataset_manifest
FILES = {"raw/b.csv": b"x,y\n1,2\n", "raw/a.csv": b"x,y\n3,4\n", "README.md": b"# data\n"}


def test_manifest_format():
    m, _ = dataset_manifest(FILES)
    lines = m.splitlines(keepends=True)
    assert len(lines) == 3 and all(line.endswith("\n") for line in lines), "파일마다 한 줄, 줄 끝은 \\n입니다"
    assert [line.split("  ")[2].strip() for line in lines] == ["README.md", "raw/a.csv", "raw/b.csv"], \
        "줄은 경로 순으로 정렬합니다"
    assert lines[1] == "f2c863cb01af6905bf817e5fb5989ab7239fe88cd81c4e11246acd573197e900  8  raw/a.csv\n", \
        "한 줄의 형식은 '<sha256 hex>  <byte 수>  <경로>'이고 구분자는 공백 두 칸입니다"


def test_known_digest():
    _, d = dataset_manifest(FILES)
    assert d == "b77ec5ff2cac22d642ea39a1e8e328024b44f35282e06cfa7148990e129d6ac8", \
        "데이터셋 해시는 manifest 문자열(UTF-8)의 sha256이어야 합니다"


def test_digest_is_hash_of_manifest():
    m, d = dataset_manifest(FILES)
    assert d == hashlib.sha256(m.encode("utf-8")).hexdigest()


def test_order_independent():
    reversed_files = dict(reversed(list(FILES.items())))
    assert dataset_manifest(reversed_files) == dataset_manifest(FILES), "딕셔너리 순서가 결과를 바꾸면 안 됩니다"


def test_detects_one_byte_change():
    changed = dict(FILES)
    changed["raw/a.csv"] = b"x,y\n3,5\n"
    assert dataset_manifest(changed)[1] != dataset_manifest(FILES)[1], "내용이 1 byte만 바뀌어도 해시가 바뀌어야 합니다"


def test_detects_rename():
    renamed = {("raw/c.csv" if k == "raw/b.csv" else k): v for k, v in FILES.items()}
    assert dataset_manifest(renamed)[1] != dataset_manifest(FILES)[1], "경로가 바뀌어도 해시가 바뀌어야 합니다"


def test_empty_file():
    m, _ = dataset_manifest({"empty.txt": b""})
    assert m == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  0  empty.txt\n"
