"""19.9 퀴즈(content/19-generative-and-foundation-models/09-foundation-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/19-generative-and-foundation-models/09-foundation-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_lora_param_count():
    d, r = 1024, 8
    A, B = np.zeros((r, d)), np.zeros((d, r))
    assert A.size + B.size == Q["lora-param-count"]["answer"] == 16384


def test_lora_init_zero():
    W0, A, B = rng.normal(size=(5, 4)), rng.normal(size=(2, 4)), np.zeros((5, 2))
    x = rng.normal(size=4)
    assert np.allclose((W0 + B @ A) @ x, W0 @ x)              # B = 0: 사전학습 함수 그대로
    # A까지 0이면 B에 대한 gradient (∂L/∂y)(A x)ᵀ가 0이 된다
    g_y = rng.normal(size=5)
    assert np.allclose(np.outer(g_y, np.zeros((2, 4)) @ x), 0)
    assert correct("lora-init-zero") == [0]


def test_negative_transfer_toy():
    p, lam = 50, 20.0
    w_src, w_t = rng.normal(size=p), rng.normal(size=p)        # 무관한 원천과 목표
    def fit(X, y, prior):
        return prior + np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ (y - X @ prior))
    X_te = rng.normal(size=(4000, p))
    def r2(w):
        y = X_te @ w_t + rng.normal(size=4000)
        return 1 - np.mean((y - X_te @ w) ** 2) / np.var(y)
    res = []
    for _ in range(20):
        X = rng.normal(size=(10, p)); y = X @ w_t + rng.normal(size=10)
        res.append([r2(fit(X, y, np.zeros(p))), r2(fit(X, y, w_src))])
    scratch, ft = np.mean(res, 0)
    assert ft < scratch and ft < 0                             # 원천에서 출발한 미세조정이 더 나쁘다
    assert correct("negative-transfer") == [1]


def test_text_answers():
    assert correct("foundation-vs-large") == [1]
    assert correct("transfer-method-small-data") == [1]
    assert correct("near-duplicate-split") == [1, 2]
