"""3.2 퀴즈(content/03-environments/02-ubuntu-and-cuda/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/03-environments/02-ubuntu-and-cuda/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def vtuple(s):
    return tuple(int(p) for p in s.split("."))


def test_driver_version_compare():
    installed, need = "525.105.17", "525.60.13"
    assert vtuple(installed) >= vtuple(need)          # 정답: 만족한다
    assert not installed >= need                      # 문자열 비교는 반대 결론(선택지 1의 오개념)
    assert correct("driver-version-compare") == [0]


def test_which_nvcc_runs():
    files = {"/usr/local/cuda-12.1/bin/nvcc", "/usr/local/cuda-12.4/bin/nvcc"}
    links = {"/usr/local/cuda": "/usr/local/cuda-12.4"}
    path = "/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/usr/bin"
    found = None
    for d in path.split(":"):
        real = d
        for src, dst in links.items():
            if d.startswith(src + "/"):
                real = dst + d[len(src):]
        if real + "/nvcc" in files:
            found = real + "/nvcc"
            break
    assert found == "/usr/local/cuda-12.1/bin/nvcc"
    assert Q["which-nvcc-runs"]["choices"][correct("which-nvcc-runs")[0]]["text"] == "12.1"


def test_other_answers():
    assert correct("os-choice-for-lab") == [0]
    assert correct("official-sources") == [0, 1, 2]
    assert correct("nvml-mismatch") == [0]
    assert correct("toolkit-only-package") == [0]


def test_every_wrong_choice_has_feedback():
    for q in Q.values():
        for c in q.get("choices", []):
            assert c.get("correct") or c.get("feedback"), f"{q['id']}: 오답에 feedback이 없다"
