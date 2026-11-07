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


dedup = load().dedup


class Key:
    """== 비교 횟수를 세는 hash 가능한 값"""

    eq_calls = 0

    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        Key.eq_calls += 1
        return isinstance(other, Key) and self.v == other.v

    def __hash__(self):
        return hash(self.v)

    def __repr__(self):
        return f"Key({self.v!r})"


def test_order_and_values():
    assert dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert dedup([]) == []
    assert dedup([3, 3, 3]) == [3]
    assert dedup(x % 4 for x in range(10)) == [0, 1, 2, 3], "generator 입력도 받아야 합니다"


def test_returns_list_and_keeps_input():
    data = [1, 2, 1]
    out = dedup(data)
    assert isinstance(out, list)
    assert data == [1, 2, 1], "입력을 바꾸면 안 됩니다"


def test_linear_number_of_comparisons():
    n = 2_000
    items = [Key(i % 1_000) for i in range(n)]       # 값 1000개가 두 번씩
    Key.eq_calls = 0
    out = dedup(items)
    assert [k.v for k in out] == list(range(1_000))
    assert Key.eq_calls <= 3 * n, (
        f"== 비교가 {Key.eq_calls:,}회 일어났습니다(원소 {n:,}개). "
        "리스트에서 membership을 검사하면 O(n^2)회가 됩니다"
    )


def test_first_occurrence_object_is_kept():
    a1, a2 = Key("a"), Key("a")
    out = dedup([a1, a2])
    assert len(out) == 1 and out[0] is a1, "처음 나온 객체를 남겨야 합니다"
