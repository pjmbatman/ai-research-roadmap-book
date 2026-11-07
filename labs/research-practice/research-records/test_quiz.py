"""23.7 퀴즈(content/23-research-practice/07-research-records/quiz.toml) 정답과 note_stamp.py를 검증한다."""
import importlib.util
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/23-research-practice/07-research-records/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

spec = importlib.util.spec_from_file_location("note_stamp", HERE / "note_stamp.py")
note_stamp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(note_stamp)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_config_hash_key_order():
    a = note_stamp.load(HERE / "example_train.toml")
    b = note_stamp.load(HERE / "example_train_reordered.toml")
    c = dict(a, lr=0.0010001)
    assert (HERE / "example_train.toml").read_bytes() != (HERE / "example_train_reordered.toml").read_bytes()
    assert note_stamp.config_hash(a) == note_stamp.config_hash(b) == "d710199e0a73"   # 본문에 적은 값
    assert note_stamp.config_hash(c) != note_stamp.config_hash(a)
    assert correct("config-hash-key-order") == [0]


def test_stamp_lines():
    out = note_stamp.stamp([HERE / "example_train.toml"])
    assert out.splitlines()[0].startswith("- 시각: ")
    assert "sha256:d710199e0a73" in out


def test_other_answers():
    assert correct("write-notes-during") == [2]
    assert correct("decision-rejected-alternatives") == [1]
    assert correct("uncommitted-run") == [0, 1]
    assert correct("weekly-report-blocker") == [2]
    assert correct("record-stopping") == [1]
