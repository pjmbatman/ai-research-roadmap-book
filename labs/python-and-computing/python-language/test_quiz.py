"""5.1 퀴즈(content/05-python-and-computing/01-python-language/quiz.toml) 정답을 실행으로 검증한다."""
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/05-python-and-computing/01-python-language/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_is_vs_equal():
    x, y = [1, 2], [1, 2]
    assert (x == y, x is y) == (True, False)
    assert correct("is-vs-equal") == [0]


def test_shallow_copy_nested():
    grid = [[0, 0], [0, 0]]
    snapshot = grid.copy()
    grid[0][0] = 99
    assert snapshot == [[99, 0], [0, 0]]
    assert correct("shallow-copy-nested") == [1]


def test_mutable_default_count():
    def collect(x, acc=[]):
        acc.append(x)
        return len(acc)

    collect(1)
    collect(2)
    assert collect(3) == Q["mutable-default-count"]["answer"]


def test_late_binding_closure():
    fs = [lambda x: x * k for k in range(3)]
    assert [f(10) for f in fs] == [20, 20, 20]
    assert correct("late-binding-closure") == [1]


def test_generator_exhausted():
    g = (x * x for x in range(4))
    assert (sum(g), sum(g)) == (14, 0)
    assert correct("generator-exhausted") == [1]


def test_type_hints_not_enforced():
    def scale(x: float) -> float:
        return 2 * x

    assert scale("ab") == "abab"
    assert correct("type-hints-not-enforced") == [1]


def test_dataclass_default_factory():
    @dataclass
    class A:
        tags: list[str] = field(default_factory=list)

    @dataclass
    class B:
        tags: list[str] | None = None

        def __post_init__(self):
            if self.tags is None:
                self.tags = []

    for cls in (A, B):
        a, b = cls(), cls()
        a.tags.append("x")
        assert b.tags == []
    for default in ("[]", "list()"):
        with pytest.raises(ValueError):
            exec(f"@dataclass\nclass C:\n    tags: list = {default}", {"dataclass": dataclass})
    assert correct("dataclass-default-factory") == [0, 1]
