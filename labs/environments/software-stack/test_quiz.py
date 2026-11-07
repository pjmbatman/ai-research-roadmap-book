"""3.1 퀴즈(content/03-environments/01-software-stack/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/03-environments/01-software-stack/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ver(s):
    return tuple(int(p) for p in s.split("."))


def test_runtime_count():
    driver = ver("12.4")
    runtimes = ["11.8", "12.1", "12.4", "12.6", "13.0"]
    assert sum(ver(r) <= driver for r in runtimes) == Q["runtime-count"]["answer"] == 3


def test_three_versions_scenario():
    driver, toolkit, torch_cuda = ver("12.4"), ver("11.8"), ver("12.1")
    assert torch_cuda <= driver                      # 선택지 0: PyTorch runtime이 드라이버 범위 안
    assert toolkit[0] != torch_cuda[0]               # 선택지 1: extension 빌드 시 major 불일치
    assert correct("three-versions-scenario") == [0, 1]


def test_single_answers():
    assert correct("nvidia-smi-cuda-version") == [0]
    assert correct("torch-wheel-needs-toolkit") == [0]
    assert correct("stack-order") == [0]
    assert correct("abi-undefined-symbol") == [0]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
