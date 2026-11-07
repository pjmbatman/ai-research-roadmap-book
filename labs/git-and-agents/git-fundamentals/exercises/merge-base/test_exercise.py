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


merge_base = load().merge_base

# A ─ B ─ C          main
#      \
#       D ─ E        feature
FORK = {"A": [], "B": ["A"], "C": ["B"], "D": ["B"], "E": ["D"]}

# criss-cross: 두 branch가 서로를 번갈아 merge했다
#   A ─ B ─ M1 ─ X      main     (M1 = merge(B, C))
#    \    ╳
#     C ─ M2 ─ Y        feature  (M2 = merge(C, B))
CRISS = {"A": [], "B": ["A"], "C": ["A"], "M1": ["B", "C"], "M2": ["C", "B"], "X": ["M1"], "Y": ["M2"]}


def test_simple_fork():
    assert merge_base(FORK, "C", "E") == ["B"], "두 branch가 갈라진 지점 B가 merge-base입니다"


def test_symmetric():
    assert merge_base(FORK, "E", "C") == merge_base(FORK, "C", "E"), "merge-base는 인자 순서와 무관해야 합니다"


def test_ancestor_is_its_own_base():
    assert merge_base(FORK, "B", "E") == ["B"], "a가 b의 조상이면 merge-base는 a 자신입니다(fast-forward 가능한 경우)"
    assert merge_base(FORK, "E", "E") == ["E"], "자기 자신과의 merge-base는 자기 자신입니다"


def test_only_best_common_ancestors():
    result = merge_base(FORK, "C", "E")
    assert "A" not in result, "A는 공통 조상이지만 다른 공통 조상 B의 조상이므로 제외해야 합니다"


def test_criss_cross_has_two_bases():
    assert merge_base(CRISS, "X", "Y") == ["B", "C"], "criss-cross 이력에서는 가장 좋은 공통 조상이 B와 C 두 개입니다"


def test_merge_commit_parents_are_followed():
    assert merge_base(CRISS, "M1", "C") == ["C"], "merge commit은 두 번째 parent도 따라가야 합니다"


def test_unrelated_histories():
    two_roots = {"A": [], "B": ["A"], "P": [], "Q": ["P"]}
    assert merge_base(two_roots, "B", "Q") == [], "공통 조상이 없으면 빈 리스트를 반환합니다"
