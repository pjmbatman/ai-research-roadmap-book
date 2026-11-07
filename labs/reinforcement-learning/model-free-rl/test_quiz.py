"""20.2 퀴즈(content/20-reinforcement-learning/02-model-free-rl/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/20-reinforcement-learning/02-model-free-rl/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_td_vs_mc():
    # 무작위 보행에서 C(가운데)의 MC 목표는 0/1(분산 1/4), TD 목표는 이웃 가치(참값이면 1/3 또는 2/3, 분산 1/36).
    v = np.arange(1, 6) / 6
    mc_var = 0.5 * 0.5
    td_targets = np.array([v[1], v[3]])
    assert np.isclose(td_targets.mean(), v[2])                 # 참 가치를 쓰면 TD 목표도 불편
    assert td_targets.var() < mc_var
    assert correct("td-vs-mc") == [0]


def test_q_update_numeric():
    q, r, g, qmax, alpha = 2.0, 1.0, 0.9, 3.0, 0.5
    new = q + alpha * (r + g * qmax - q)
    assert abs(new - Q["q-update-numeric"]["answer"]) <= Q["q-update-numeric"]["tolerance"]


def test_q_vs_sarsa_targets_differ():
    Qs = np.array([0.0, -100.0])             # 다음 상태: 행동 0은 안전, 행동 1은 절벽
    eps = 0.1
    q_target = Qs.max()
    sarsa_expected = (1 - eps + eps / 2) * Qs[0] + (eps / 2) * Qs[1]   # ε-greedy가 실제로 고를 행동의 기댓값
    assert q_target > sarsa_expected                         # SARSA는 탐험 위험을 반영
    assert correct("q-vs-sarsa-cliff") == [0]


def test_baseline_unbiased():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=3)
    pi = np.exp(theta) / np.exp(theta).sum()
    score = np.eye(3) - pi                   # 행 a = grad log pi(a)
    assert np.allclose(pi @ score, 0)        # E[score] = 0 -> 상수(상태 의존) baseline은 불편
    qa = np.array([1.0, 2.0, 3.0])           # 행동 의존 baseline b(a) = Q(a)
    assert not np.allclose(pi @ (qa[:, None] * score), 0)    # 편향 발생
    assert correct("baseline-unbiased") == [0, 1]


def test_ppo_clip_limits_update():
    eps, adv = 0.2, 1.0
    obj = lambda rho: min(rho * adv, np.clip(rho, 1 - eps, 1 + eps) * adv)
    h = 1e-6
    assert np.isclose((obj(1.0 + h) - obj(1.0)) / h, adv)          # 안쪽에서는 기울기 있음
    assert np.isclose((obj(1.5 + h) - obj(1.5)) / h, 0.0)          # 1 + eps 밖에서는 더 키워도 이득 없음
    assert correct("ppo-clip") == [0]


def test_seed_reporting_interval():
    a = np.array([int(c) for c in "001110011011101110011111011110"], dtype=float)   # 본문 cell 4 출력
    b = np.array([int(c) for c in "110011110000101000011111000110"], dtype=float)
    assert round(a.mean(), 2) == 0.67 and round(b.mean(), 2) == 0.50
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    lo, hi = a.mean() - b.mean() - 1.96 * se, a.mean() - b.mean() + 1.96 * se
    assert (round(lo, 2), round(hi, 2)) == (-0.08, 0.42)
    assert lo < 0 < hi and a[0] == 0 and b[0] == 1
    assert correct("seed-reporting") == [2]
