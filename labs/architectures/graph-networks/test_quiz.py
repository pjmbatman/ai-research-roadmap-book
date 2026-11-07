"""18.6 퀴즈(content/18-architectures/06-graph-networks/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/06-graph-networks/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_gcn_normalized_weight():
    # 노드 0: 차수 3 (1, 2, 3과 연결), 노드 1: 차수 2 (0, 4와 연결)
    A = np.zeros((5, 5))
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 4)]:
        A[i, j] = A[j, i] = 1
    A_hat = A + np.eye(5)
    d = A_hat.sum(1)
    w = (A_hat / np.sqrt(np.outer(d, d)))[0, 1]
    q = Q["gcn-normalized-weight"]
    assert abs(w - q["answer"]) <= q["tolerance"]
    assert abs(1 / np.sqrt(3 * 2) - q["answer"]) > q["tolerance"]      # self-loop 누락 오답과 구별


def test_receptive_field_hops():
    n = 10
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    reach = np.eye(n)
    layers = 0
    while reach[0, n - 1] == 0:                                        # 노드 1(인덱스 0)이 노드 10을 볼 때까지
        reach = ((A + np.eye(n)) @ reach > 0).astype(float)
        layers += 1
    assert layers == Q["receptive-field-hops"]["answer"] == 9


def test_mean_cannot_count():
    x = np.array([1.0, 0.0, 0.0])
    one, two = np.stack([x]), np.stack([x, x])
    aggs = [lambda M: M.sum(0), lambda M: M.mean(0), lambda M: M.max(0)]
    distinguishes = [not np.allclose(f(one), f(two)) for f in aggs]
    assert distinguishes == [True, False, False]
    assert correct("mean-cannot-count") == [0]


def test_node_vs_graph_target():
    N = 6
    A = np.triu(rng.random((N, N)) < 0.5, 1).astype(float); A = A + A.T
    X = rng.normal(size=(N, 3)); W1, W2 = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    f = lambda A, X: np.tanh(X @ W1 + A @ X @ W2)                      # 노드 출력
    P = np.eye(N)[rng.permutation(N)]
    assert np.allclose(f(P @ A @ P.T, P @ X), P @ f(A, X))             # 등변
    assert np.allclose(f(P @ A @ P.T, P @ X).sum(0), f(A, X).sum(0))   # readout 불변
    assert correct("node-vs-graph-target") == [0]


def test_complete_graph_mean():
    S = 30
    Y = rng.normal(size=(50, S))
    G = 1.0 - np.eye(S)
    pred = Y @ G.T / G.sum(1)
    loo_mean = (Y.sum(1, keepdims=True) - Y) / (S - 1)                 # 자기를 뺀 전역 평균
    assert np.allclose(pred, loo_mean)
    assert correct("complete-graph-mean") == [0]


def test_oversmoothing_vs_squashing():
    n = 20
    A = np.triu(rng.random((n, n)) < 0.3, 1).astype(float); A = A + A.T
    A_hat = A + np.eye(n); d = A_hat.sum(1)
    A_norm = A_hat / np.sqrt(np.outer(d, d))
    H = rng.normal(size=(n, 4))
    energy = lambda H: 0.5 * (A * ((H[:, None] - H[None]) ** 2).sum(-1)).sum()
    H2 = np.linalg.matrix_power(A_norm, 2) @ H
    H16 = np.linalg.matrix_power(A_norm, 16) @ H
    Hn = lambda H: H / np.sqrt(d)[:, None]                             # 차수 크기 효과 제거
    assert energy(Hn(H16)) < energy(Hn(H2))
    assert correct("oversmoothing-vs-squashing") == [0]
