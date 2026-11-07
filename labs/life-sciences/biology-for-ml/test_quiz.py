"""29.1 퀴즈(content/29-life-sciences/01-biology-for-ml/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid

QUIZ = Path(__file__).parents[3] / "content/29-life-sciences/01-biology-for-ml/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

BASES = "TCAG"
AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: AA[16 * i + 4 * j + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_genetic_code_table():
    assert CODON["ATG"] == "M" and CODON["TGG"] == "W"
    assert sorted(k for k, v in CODON.items() if v == "*") == ["TAA", "TAG", "TGA"]
    assert len(set(AA) - {"*"}) == 20


def test_synonymous_variant():
    assert CODON["GCT"] == CODON["GCC"] == "A"
    assert CODON["GCC"] != "*"
    assert correct("synonymous-variant") == [1]


def test_central_dogma_flow():
    assert correct("central-dogma-flow") == [2]


def test_coding_length():
    n_aa = 100
    assert n_aa * 3 + 3 == Q["coding-length"]["answer"] == 303


def test_tissue_difference():
    assert correct("tissue-difference") == [1]


def test_technical_vs_biological():
    """깊이를 늘리면 기술적 반복의 상관은 1에 다가가지만 생물학적 반복은 그렇지 않다."""
    rng = np.random.default_rng(1)
    frac = rng.lognormal(0, 1.5, 2000)
    frac /= frac.sum()
    bio = frac * rng.lognormal(0, 0.3, 2000)
    bio /= bio.sum()
    depth = 1_000_000
    r = lambda a, b: np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]
    tech = r(rng.poisson(depth * frac), rng.poisson(depth * frac))
    biol = r(rng.poisson(depth * frac), rng.poisson(depth * bio))
    assert tech > 0.99 and biol < tech - 0.01
    assert correct("technical-vs-biological") == [0]


def test_confounded_design():
    """조건 효과가 0이어도 배치와 겹치면 분류 정확도가 1에 가깝다."""
    rng = np.random.default_rng(2)
    cond = np.repeat([0, 1], 20)
    X = rng.normal(0, 1, (40, 200)) + np.outer(cond, rng.normal(0, 0.5, 200))   # 배치 = 조건
    assert cross_val_score(NearestCentroid(), X, cond, cv=5).mean() > 0.95
    # 배치를 제거(배치별 평균 빼기)하면 조건 신호도 함께 사라진다
    Xc = X - np.where(cond[:, None] == 1, X[cond == 1].mean(0), X[cond == 0].mean(0))
    assert np.allclose(Xc[cond == 1].mean(0), Xc[cond == 0].mean(0))
    assert correct("confounded-design") == [1]
