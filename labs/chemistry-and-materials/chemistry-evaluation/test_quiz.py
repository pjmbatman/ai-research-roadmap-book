"""28.5 퀴즈(content/28-chemistry-and-materials/05-chemistry-evaluation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/28-chemistry-and-materials/05-chemistry-evaluation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def tanimoto(A, B):
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def test_near_duplicate_leakage():
    assert correct("near-duplicate-leakage") == [0]
    # 근접 중복이 있으면 1-NN 암기 baseline의 오차가 작아짐을 확인
    rng = np.random.default_rng(0)
    L = 48
    X = (rng.random((400, L)) < 0.15).astype(bool)
    X[:100] = X[200:300]                                   # test 앞 100개를 train 복사본으로
    w = rng.normal(size=L)
    y = X.astype(float) @ w
    tr, te = np.arange(200, 400), np.arange(0, 200)
    S = tanimoto(X[te], X[tr])
    nn = np.argmax(S, 1)
    err = np.abs(y[tr][nn] - y[te])
    leaked = S.max(1) >= 0.99
    assert err[leaked].mean() < err[~leaked].mean()        # 누수 분자에서 오차가 작다


def test_mixed_levels():
    assert correct("mixed-levels") == [0]
    # 체계적 offset이 섞이면 단일 수준보다 오차가 커짐
    rng = np.random.default_rng(1)
    L = 32
    X = (rng.random((600, L)) < 0.2).astype(float)
    w = rng.normal(size=L)
    level = rng.integers(0, 2, 600)
    y = X @ w + np.array([0.0, 3.0])[level]
    def fit_mae(idx_tr, idx_te):
        A = np.c_[X[idx_tr], np.ones(len(idx_tr))]
        beta = np.linalg.solve(A.T @ A + np.eye(L + 1), A.T @ y[idx_tr])
        return np.abs(np.c_[X[idx_te], np.ones(len(idx_te))] @ beta - y[idx_te]).mean()
    tr, te = np.arange(100, 600), np.arange(100)
    mae_mixed = fit_mae(tr, te)
    m0 = level == 0
    mae_single = fit_mae(tr[m0[tr]], te[m0[te]])
    assert mae_mixed > 3 * mae_single


def test_error_bars():
    assert correct("error-bars") == [0]


def test_benchmark_audit():
    assert correct("benchmark-audit") == [0, 1, 2]
