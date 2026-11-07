"""3.3 퀴즈(content/03-environments/03-python-environments/quiz.toml) 정답을 확인한다."""
import sys
import tempfile
import tomllib
import venv
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/03-environments/03-python-environments/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_venv_reuses_existing_interpreter():
    # interpreter-managers: venv는 새 인터프리터를 설치하지 않고 기존 것을 가리킨다
    with tempfile.TemporaryDirectory() as tmp:
        venv.create(tmp, with_pip=False, symlinks=True)
        cfg = dict(line.split(" = ", 1) for line in (Path(tmp) / "pyvenv.cfg").read_text().splitlines() if " = " in line)
        assert Path(cfg["home"]).resolve() == Path(sys._base_executable).resolve().parent
        assert cfg["version"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    texts = [Q["interpreter-managers"]["choices"][i]["text"] for i in correct("interpreter-managers")]
    assert texts == ["uv", "conda", "pyenv"]


def test_other_answers():
    assert correct("conda-pip-mixing") == [0]
    assert correct("pip-python-mismatch") == [0]
    assert correct("externally-managed") == [0]
    assert correct("base-environment") == [0]
    assert correct("lab-standard-rules") == [0, 1, 2]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
