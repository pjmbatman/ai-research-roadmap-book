"""21.4 퀴즈(content/21-reliability/04-distribution-shift/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, norm

QUIZ = Path(__file__).parents[3] / "content/21-reliability/04-distribution-shift/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_label_shift_changes_posterior_not_likelihood():
    # p(x|y): 정상 N(0,1), 결함 N(2,1). 결함률 2% → 10%
    x = 1.0
    post = lambda prior: prior * norm.pdf(x, 2, 1) / (prior * norm.pdf(x, 2, 1) + (1 - prior) * norm.pdf(x, 0, 1))
    assert post(0.10) > 2 * post(0.02)                  # p(y|x)는 바뀐다
    # 사전확률 재가중으로 정확히 복원된다
    pa = post(0.02)
    odds = pa / (1 - pa) * (0.10 / 0.90) / (0.02 / 0.98)
    assert np.isclose(odds / (1 + odds), post(0.10))
    assert correct("shift-type-sensor") == [0]


def fit_line(x, y, w):
    A = np.stack([np.ones_like(x), x], 1)
    return np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))


def test_iw_helps_misspecified_not_wellspecified():
    xs, xt = rng.normal(0, 1, 3000), rng.normal(1.5, 0.6, 50000)
    w = norm.pdf(xs, 1.5, 0.6) / norm.pdf(xs, 0, 1)
    mse = lambda c, f: np.mean((c[0] + c[1] * xt - f(xt)) ** 2)
    curved = lambda x: x + 0.3 * x ** 2
    ys = curved(xs) + rng.normal(0, 0.3, xs.size)
    assert mse(fit_line(xs, ys, w), curved) < 0.3 * mse(fit_line(xs, ys, np.ones_like(xs)), curved)
    straight = lambda x: 1 + 2 * x                     # 잘 지정된 경우: 둘 다 참 직선 근처
    ys2 = straight(xs) + rng.normal(0, 0.3, xs.size)
    assert mse(fit_line(xs, ys2, np.ones_like(xs)), straight) < 0.01
    assert correct("iw-when-needed") == [0]


def test_ood_detection_vs_generalization():
    assert correct("ood-detection-vs-generalization") == [0]


def test_marginal_ks_misses_correlation_change():
    L = np.array([[1.0, 0.0], [0.8, 0.6]])
    rej = []
    for _ in range(200):
        A, B = rng.normal(size=(400, 2)), rng.normal(size=(400, 2)) @ L.T
        rej.append(min(ks_2samp(A[:, j], B[:, j]).pvalue for j in range(2)) * 2 < 0.05)
    assert np.mean(rej) < 0.12                         # 기각률이 유의수준 근처
    # 상호작용 특징의 평균은 크게 다르다 (분류기가 쓸 수 있는 신호)
    assert abs(np.mean(B[:, 0] * B[:, 1]) - np.mean(A[:, 0] * A[:, 1])) > 0.5
    assert correct("marginal-tests-miss") == [0]


def test_monitoring_false_alarms():
    p = rng.uniform(size=(20000, 20))                  # 귀무가설 아래 p-value는 균등분포
    assert abs((p < 0.05).sum(1).mean() - 1.0) < 0.02
    assert Q["monitoring-false-alarms"]["answer"] == 20 * 0.05


def test_worst_group():
    a_b = (0.95 - 0.95 * 0.97) / 0.05
    assert np.isclose(a_b, 0.57)
    assert [Q["worst-group"]["choices"][i]["text"] for i in correct("worst-group")] == ["0.57"]
