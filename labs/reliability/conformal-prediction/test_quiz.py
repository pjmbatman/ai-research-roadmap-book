"""21.3 퀴즈(content/21-reliability/03-conformal-prediction/quiz.toml) 정답을 계산으로 검증한다."""
import math
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/21-reliability/03-conformal-prediction/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def qhat(scores, alpha):
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return np.sort(scores)[k - 1] if k <= len(scores) else np.inf


def coverage(n, alpha, shift=False, reps=400, noise_scale=1.0):
    """보정 n개로 절댓값 점수 구간을 만들고 시험 포함률을 반복 평균한다. 모델은 일부러 틀린 0.8 x."""
    out = []
    for _ in range(reps):
        xc = rng.uniform(0, 3, n)
        xt = 3 * rng.beta(4, 1, 300) if shift else rng.uniform(0, 3, 300)
        yc = np.sin(xc) + rng.standard_t(3, n) * (0.1 + 0.3 * xc)
        yt = np.sin(xt) + rng.standard_t(3, 300) * (0.1 + 0.3 * xt) * noise_scale
        q = qhat(np.abs(yc - 0.8 * xc), alpha)
        out.append(np.mean(np.abs(yt - 0.8 * xt) <= q))
    return np.mean(out)


def test_marginal_holds_but_conditional_fails():
    xc = rng.uniform(0, 3, 2000)
    yc = np.sin(xc) + rng.normal(size=2000) * (0.05 + 0.3 * xc)
    q = qhat(np.abs(yc - np.sin(xc)), 0.1)
    xt = rng.uniform(0, 3, 50000)
    inside = np.abs(rng.normal(size=50000) * (0.05 + 0.3 * xt)) <= q
    assert abs(inside.mean() - 0.9) < 0.01                  # 주변 포함률은 맞다
    assert inside[xt > 2.5].mean() < 0.8                    # 특정 x에서는 크게 모자란다
    assert correct("marginal-not-conditional") == [0]


def test_exchangeability_breaks():
    assert coverage(50, 0.1) >= 0.89                        # t 잡음 + 틀린 모델: 보장 유지
    assert coverage(50, 0.1, shift=True) < 0.85             # 공변량 이동: 깨짐
    assert coverage(50, 0.1, noise_scale=2.0) < 0.85        # 잡음 증가(drift): 깨짐
    assert correct("exchangeability-breaks") == [0, 1]


def test_conformal_rank():
    assert math.ceil(20 * 0.9) == Q["conformal-rank"]["answer"] == 18
    assert (math.ceil(11 * 0.9), math.ceil(10 * 0.9)) == (10, 9)


def test_too_few_calibration():
    assert qhat(np.arange(8.0), 0.1) == np.inf
    assert np.isclose(8 / 9, 0.8889, atol=1e-4)             # 최대 점수를 써도 8/9 < 0.9
    assert np.quantile(np.arange(8.0), 0.9) < 7.0           # np.quantile은 최대 점수보다 작은 값
    assert correct("too-few-calibration") == [0]


def test_normalized_score_keeps_marginal_and_evens_out():
    sig = lambda x: 0.05 + 0.3 * x
    xc, xt = rng.uniform(0, 3, 3000), rng.uniform(0, 3, 50000)
    rc, rt = rng.normal(size=3000) * sig(xc), rng.normal(size=50000) * sig(xt)
    q = qhat(np.abs(rc) / sig(xc), 0.1)
    inside = np.abs(rt) <= q * sig(xt)
    assert abs(inside.mean() - 0.9) < 0.01
    assert abs(inside[xt > 2.5].mean() - 0.9) < 0.02 and abs(inside[xt < 0.5].mean() - 0.9) < 0.02
    assert correct("normalized-score-effect") == [0]
