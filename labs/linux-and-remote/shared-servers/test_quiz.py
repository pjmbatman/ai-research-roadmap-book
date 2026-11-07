"""2.5 퀴즈(content/02-linux-and-remote/05-shared-servers/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/02-linux-and-remote/05-shared-servers/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_cuda_visible_renumber():
    visible = [2, 3]                       # CUDA_VISIBLE_DEVICES=2,3
    assert visible[0] == 2                 # cuda:0 → 물리 GPU 2
    assert correct("cuda-visible-renumber") == [0]


def test_array_concurrency():
    spec = "0-9%3"
    rng, limit = spec.split("%")
    lo, hi = map(int, rng.split("-"))
    assert hi - lo + 1 == 10
    assert int(limit) == Q["array-concurrency"]["answer"]


def test_checkpoint_interval_minutes():
    c_min, m_min = 3, 8 * 60
    tau = math.sqrt(2 * c_min * m_min)
    q = Q["checkpoint-interval-minutes"]
    assert abs(tau - q["answer"]) <= q["tolerance"]
    assert abs(math.sqrt(2 * 3 * 8) - q["answer"]) > q["tolerance"]   # 단위를 섞은 오답


def test_checkpoint_interval_matches_simulation():
    # cells.py cell 3과 같은 모형: 근사 최적 주기 근처가 아주 짧거나 긴 주기보다 빠르다
    def wall(interval_h, rng, work_h=10.0, ckpt_h=0.05, mtbf_h=8.0, restart_h=0.1):
        done, t = 0.0, 0.0
        while done < work_h:
            seg = min(interval_h, work_h - done)
            fail_in = rng.exponential(mtbf_h)
            if fail_in >= seg + ckpt_h:
                t, done = t + seg + ckpt_h, done + seg
            else:
                t += fail_in + restart_h
        return t

    rng = np.random.default_rng(1)
    mean = {i: np.mean([wall(i, rng) for _ in range(1500)]) for i in [0.25, 0.9, 4.0]}
    assert mean[0.9] < mean[0.25] and mean[0.9] < mean[4.0]


def test_choice_answers():
    assert correct("login-node-compute") == [0]
    assert correct("scratch-not-backed-up") == [0]
    assert correct("shared-etiquette") == [0, 1, 2]
