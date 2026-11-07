"""3.4 퀴즈(content/03-environments/04-dependencies-and-packaging/quiz.toml) 정답을 계산으로 검증한다."""
import importlib.util
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
QUIZ = Path(__file__).parents[3] / "content/03-environments/04-dependencies-and-packaging/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

_spec = importlib.util.spec_from_file_location("vs_solution", HERE / "exercises/version-specifier/solution.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
satisfies = _mod.satisfies


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_compatible_release():
    texts = [c["text"] for c in Q["compatible-release"]["choices"]]
    picks = [i for i, t in enumerate(texts) if satisfies(t, "~=1.4.2")]
    assert picks == correct("compatible-release") == [0]
    assert satisfies("1.5.0", "~=1.4")                      # 해설의 ~=1.4 설명


def test_transitive_count():
    meta = {"proj": ["pandas", "scipy", "matplotlib"], "pandas": ["numpy", "python-dateutil"],
            "scipy": ["numpy"], "matplotlib": ["numpy", "pillow"], "python-dateutil": ["six"]}
    seen, stack = set(), list(meta["proj"])
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack += meta.get(p, [])
    assert len(seen - set(meta["proj"])) == Q["transitive-count"]["answer"] == 4


def test_read_resolver_error():
    # pandas 3.1.1의 요구(numpy>=2.0.2)와 프로젝트의 numpy<2를 동시에 만족하는 버전이 없다
    candidates = [f"{a}.{b}.{c}" for a in (1, 2) for b in range(0, 27) for c in range(0, 6)]
    assert not [v for v in candidates if satisfies(v, ">=2.0.2,<2")]
    assert correct("read-resolver-error") == [0]


def test_unexpected_source_build():
    # 집필 시점 PyPI의 numpy 1.26.4 wheel이 지원하는 Python tag (pypi.org JSON API로 확인)
    numpy_1_26_4_tags = {"cp39", "cp310", "cp311", "cp312", "pp39"}
    assert "cp313" not in numpy_1_26_4_tags
    assert correct("unexpected-source-build") == [0]


def test_other_answers():
    assert correct("freeze-as-declaration") == [0]
    assert correct("lockfile-facts") == [0, 1, 2]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
