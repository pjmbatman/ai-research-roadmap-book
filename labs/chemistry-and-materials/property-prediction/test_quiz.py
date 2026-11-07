"""28.2 퀴즈(content/28-chemistry-and-materials/02-property-prediction/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/28-chemistry-and-materials/02-property-prediction/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def tanimoto(A, B):
    inter = A.astype(float) @ B.T.astype(float)
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter)


def test_noise_floor():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.5, 200000)
    mae = np.abs(0.5 * rng.normal(size=y.size)).mean()        # 완벽한 모델의 MAE = E|잡음|
    assert abs(mae - 0.5 * np.sqrt(2 / np.pi)) < 0.01
    assert abs(Q["noise-floor"]["answer"] - 0.5 * np.sqrt(2 / np.pi)) <= Q["noise-floor"]["tolerance"]


def test_random_split_optimistic():
    # 무작위 분할의 test 최근접 유사도 > scaffold 분할의 것 (쉬운 보간)
    rng = np.random.default_rng(1)
    n_scaf, n_mol = 60, 800
    scaf = rng.choice(n_scaf, n_mol)
    bits = [rng.choice(128, 12, replace=False) for _ in range(n_scaf)]
    X = np.zeros((n_mol, 128), dtype=bool)
    for i, s in enumerate(scaf):
        X[i, bits[s]] = True
        X[i, rng.choice(128, 3, replace=False)] = True
    perm = rng.permutation(n_mol)
    r_tr, r_te = perm[160:], perm[:160]
    test_s = set(rng.permutation(n_scaf)[:12].tolist())
    s_te = np.array([i for i in range(n_mol) if scaf[i] in test_s])
    s_tr = np.array([i for i in range(n_mol) if scaf[i] not in test_s])
    sim_rand = np.median(tanimoto(X[r_te], X[r_tr]).max(1))
    sim_scaf = np.median(tanimoto(X[s_te], X[s_tr]).max(1))
    assert sim_rand > sim_scaf
    assert correct("random-split-optimistic") == [0]


def test_applicability_domain():
    assert correct("applicability-domain") == [0, 1, 3]


def test_split_choice():
    assert correct("split-choice") == [0]
