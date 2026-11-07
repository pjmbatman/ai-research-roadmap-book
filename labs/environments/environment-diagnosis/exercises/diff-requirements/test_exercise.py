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


diff_requirements = load().diff_requirements

BEFORE = """\
numpy==2.1.3
pandas==2.2.3
python-dateutil==2.9.0.post0
six==1.16.0
"""
AFTER = """\
numpy==2.2.0
pandas==2.2.3
python-dateutil==2.9.0.post0
tzdata==2024.2
"""


def test_added_removed_changed():
    r = diff_requirements(BEFORE, AFTER)
    assert r["changed"] == {"numpy": ("2.1.3", "2.2.0")}
    assert r["added"] == {"tzdata": "2024.2"}
    assert r["removed"] == {"six": "1.16.0"}


def test_unchanged_is_not_reported():
    r = diff_requirements(BEFORE, BEFORE)
    assert r == {"added": {}, "removed": {}, "changed": {}}, "같은 목록이면 세 dict 모두 비어 있어야 합니다"


def test_comments_options_and_blank_lines():
    before = "# 업그레이드 전\n\nnumpy==2.1.3\n--index-url https://example.org/simple\n"
    after = "  numpy==2.1.3  \n-e git+https://example.org/repo.git#egg=tool\n# 끝\n"
    assert diff_requirements(before, after) == {"added": {}, "removed": {}, "changed": {}}, (
        "빈 줄, # 주석, -로 시작하는 옵션 줄은 무시하고, 줄 앞뒤 공백은 지웁니다"
    )


def test_name_normalization():
    r = diff_requirements("Pillow==10.4.0\nruamel.yaml==0.18.6\n", "pillow==11.0.0\nruamel-yaml==0.18.6\n")
    assert r["changed"] == {"pillow": ("10.4.0", "11.0.0")}, "Pillow와 pillow는 같은 패키지입니다"
    assert r["added"] == {} and r["removed"] == {}, "ruamel.yaml과 ruamel-yaml은 같은 패키지입니다"


def test_direct_reference():
    r = diff_requirements("mypkg @ file:///home/me/mypkg\n", "mypkg==0.2.0\n")
    assert r["changed"] == {"mypkg": ("file:///home/me/mypkg", "0.2.0")}, "'이름 @ 주소' 줄은 주소를 버전 자리에 씁니다"
