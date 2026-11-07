"""29.4 퀴즈(content/29-life-sciences/04-omics-and-single-cell/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/29-life-sciences/04-omics-and-single-cell/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def within(qid, value):
    return abs(value - Q[qid]["answer"]) <= Q[qid]["tolerance"]


def test_poisson_zeros():
    assert within("poisson-zeros", np.exp(-0.1))
    sim = np.mean(rng.poisson(0.1, 200_000) == 0)
    assert within("poisson-zeros", sim)


def test_normalized_value():
    assert within("normalized-value", np.log1p(10 / 2000 * 1e4))
    assert np.isclose(np.log1p(10 / 2000 * 1e4), np.log1p(20 / 4000 * 1e4))   # 깊이가 두 배여도 같은 값


def test_log1p_zero():
    assert np.log1p(0.0) == 0.0
    assert correct("log1p-zero") == [0]


def test_pc1_library_size():
    """정규화하지 않은 개수의 첫 주성분은 총 개수를 따라간다."""
    base = rng.lognormal(0, 1, 200)
    size = rng.lognormal(0, 0.6, 300)
    C = rng.poisson(size[:, None] * base[None, :])
    Z = C - C.mean(0)
    pc1 = np.linalg.svd(Z, full_matrices=False)[0][:, 0]
    assert abs(np.corrcoef(pc1, C.sum(1))[0, 1]) > 0.9
    assert correct("pc1-library-size") == [0]


def test_overcorrection():
    """배치 평균 빼기는 배치에만 있는 반응을 줄이고 다른 유형에 가짜 반응을 만든다."""
    n0, n2 = 40, 160                                   # 배치 2의 유형 0, 유형 2 세포 수
    resp = np.r_[np.zeros(n0), np.full(n2, 1.0)]       # 유형 2만 반응(로그 공간 +1)
    base_b1 = np.zeros(400)                            # 배치 1(반응 없음), 같은 구성이라 가정
    b2 = resp - resp.mean()                            # 배치 2 평균을 뺀 결과
    assert np.isclose(b2[n0:].mean() - base_b1.mean(), 1.0 * (1 - n2 / (n0 + n2)))   # 반응이 줄어듦
    assert b2[:n0].mean() < 0                          # 반응하지 않은 유형 0에 가짜 음의 반응
    assert correct("overcorrection") == [1]


def test_batch_cluster_evidence():
    assert correct("batch-cluster-evidence") == [0, 1]
