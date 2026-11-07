"""6.1 퀴즈(content/06-research-code-engineering/01-project-structure/quiz.toml) 정답을 계산으로 검증한다."""
import hashlib
import json
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/06-research-code-engineering/01-project-structure/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_notebook_hidden_state():
    ns = {}
    exec("scale = 2.0", ns)                                  # 셀 1
    exec("def normalize(v):\n    return v / scale", ns)      # 셀 2
    exec("scale = 4.0", ns)                                  # 셀 3
    out = eval("normalize(8.0)", ns)                          # 셀 4
    assert out == Q["notebook-hidden-state"]["answer"] == 2.0


def test_run_id_config_hash():
    def h(cfg):
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

    base = {"model": {"width": 64}, "train": {"lr": 0.001, "seed": 0}}
    choices = [
        {"train": {"seed": 0, "lr": 0.001}, "model": {"width": 64}},
        {"model": {"width": 64}, "train": {"lr": 0.001, "seed": 1}},
        {"model": {"width": 64.0}, "train": {"lr": 0.001, "seed": 0}},
        {"model": {"width": 64}, "train": {"lr": "0.001", "seed": 0}},
    ]
    same = [i for i, c in enumerate(choices) if h(c) == h(base)]
    assert same == correct("run-id-config-hash") == [0]


def test_single_answer_structure():
    assert correct("eval-independent") == [1]
    assert correct("what-goes-to-src") == [0, 1]
    assert correct("template-benefit") == [1]
    assert correct("premature-abstraction") == [2]
