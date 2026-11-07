"""1.4 퀴즈(content/01-research-landscape/04-modeling-axes/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import beta, multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

QUIZ = Path(__file__).parents[3] / "content/01-research-landscape/04-modeling-axes/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_beta_posterior_mean():
    q = Q["beta-posterior-mean"]
    post = beta(1 + 3, 1 + 0)
    assert abs(post.mean() - q["answer"]) <= q["tolerance"]
    lo, hi = post.ppf([0.025, 0.975])
    assert round(lo, 3) == 0.398 and round(hi, 3) == 0.994     # 해설에 쓴 구간


def test_far_point_confidence():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.vstack([rng.multivariate_normal([0, 0], cov, 300), rng.multivariate_normal([2, 1], cov, 300)])
    y = np.repeat([0, 1], 300)
    far = np.array([[12.0, 6.0]])
    assert LogisticRegression().fit(X, y).predict_proba(far)[0, 1] > 0.999
    gen = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
    p_x = sum(gen.priors_[k] * multivariate_normal(gen.means_[k], gen.covariance_).pdf(far) for k in (0, 1))
    assert p_x < 1e-20
    assert correct("far-point-confidence") == [1]


def test_generative_posterior_is_logistic_form():
    """공유 공분산 가우시안 생성 모델의 log-odds는 x의 1차 함수다(axes-independent의 근거)."""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(1.5, 1, (100, 2))])
    y = np.repeat([0, 1], 100)
    gen = LinearDiscriminantAnalysis().fit(X, y)
    t = np.linspace(-3, 3, 7)[:, None] * np.array([[1.0, -0.5]])   # 직선 위의 점들
    p = gen.predict_proba(t)[:, 1]
    logit = np.log(p / (1 - p))
    assert np.allclose(np.diff(logit, 2), 0, atol=1e-8)
    assert correct("axes-independent") == [0, 1]


def test_single_choice_answers():
    assert correct("learning-signal") == [1]
    assert correct("training-configuration") == [2]
