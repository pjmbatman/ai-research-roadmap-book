"""22.3 퀴즈(content/22-experiment-operations/03-experiment-automation/quiz.toml) 정답을 계산으로 검증한다."""
import hashlib
import itertools
import json
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/03-experiment-automation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_manifest_job_id():
    job_id = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]
    a = [{"lr": lr, "seed": s} for lr, s in itertools.product([1e-3, 1e-2], [0, 1])]
    b = [{"lr": lr, "seed": s} for lr, s in itertools.product([1e-4, 1e-2, 1e-3], [0, 1])]   # 값 추가, 순서 바꿈
    assert {job_id(c) for c in a} <= {job_id(c) for c in b}         # hash ID는 유지된다
    assert any(a[i] != b[i] for i in range(len(a)))                 # 순번 ID는 다른 설정을 가리킨다
    assert correct("manifest-job-id") == [0]


def test_grid_vs_random():
    grid_lrs = np.unique(np.meshgrid(np.linspace(-5, -1, 3), np.linspace(-6, -3, 3))[0])
    assert len(grid_lrs) == 3
    rng = np.random.default_rng(0)
    opt = rng.uniform(-5, -1, 4000)
    grid_gap = np.min(np.abs(opt[:, None] - grid_lrs[None, :]), axis=1)            # (4000,)
    rand_gap = np.min(np.abs(opt[:, None] - rng.uniform(-5, -1, (4000, 9))), axis=1)
    assert np.mean(rand_gap) < np.mean(grid_gap)
    assert correct("grid-vs-random") == [0]


def test_sh_budget():
    n, budget, total = 27, 1, 0
    while True:
        total += n * budget
        if n == 1:
            break
        n, budget = max(1, n // 3), budget * 3
    assert total == Q["sh-budget"]["answer"] == 108
    assert 27 * 27 == 729 and 108 // 27 == 4                         # 같은 예산의 random search는 4개


def test_late_bloomer():
    curve = lambda final, k, t: final + (1 - final) * np.exp(-k * t)
    fast, slow = (0.30, 1.0), (0.10, 0.05)                             # 작은 lr: 느리지만 최종이 더 좋다
    assert curve(*slow, 1) > curve(*fast, 1) and curve(*slow, 200) < curve(*fast, 200)
    assert correct("late-bloomer") == [0]


def test_choice_keys():
    assert correct("fair-budget") == [0]
    assert correct("atomic-save") == [0]
    assert correct("which-to-retry") == [0, 1]
