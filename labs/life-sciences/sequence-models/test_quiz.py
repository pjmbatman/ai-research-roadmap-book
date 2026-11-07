"""29.2 퀴즈(content/29-life-sciences/02-sequence-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from itertools import product
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/29-life-sciences/02-sequence-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
AA = "ACDEFGHIKLMNPQRSTVWY"


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_kmer_dimension():
    assert len(list(product(AA, repeat=4))) == Q["kmer-dimension"]["answer"] == 160000


def test_composition_only():
    a, b = "MKVLA", "ALVKM"
    assert sorted(a) == sorted(b) and a != b                  # 조성은 같고 순서는 다르다
    assert correct("composition-only") == [0]


def test_mlm_objective():
    assert correct("mlm-objective") == [0]


def test_coevolution_conserved_column_has_zero_mi():
    """완전히 보존된 열과 다른 열의 상호정보량은 0이다(선택지 2가 틀린 이유)."""
    rng = np.random.default_rng(0)
    x = np.zeros(500, dtype=int)
    z = rng.integers(0, 20, 500)
    joint = np.zeros((20, 20))
    np.add.at(joint, (x, z), 1)
    joint /= joint.sum()
    px, pz = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    assert abs(np.sum(joint[nz] * np.log(joint[nz] / (px @ pz)[nz]))) < 1e-12
    assert correct("coevolution-signal") == [1]


def test_split_reverses_ranking():
    # 본문 cell 4의 결과: (무작위, 가족)
    knn, logreg = (1.00, 0.63), (0.84, 0.76)
    assert knn[0] > logreg[0] and logreg[1] > knn[1]
    assert correct("split-reverses-ranking") == [1]


def test_pretrain_claim_checks():
    assert correct("pretrain-claim-checks") == [0, 1, 2]
