"""26.3 퀴즈(content/26-physics-and-engineering/03-turbulence-and-closure/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/26-physics-and-engineering/03-turbulence-and-closure/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def tau_block(u, K):
    b = u.reshape(-1, K)
    return 0.5 * ((b**2).mean(1) - b.mean(1) ** 2), b.mean(1)


def test_filtered_product_stress():
    tau, _ = tau_block(np.array([1.0, 3.0]), 2)
    assert np.isclose(tau[0], Q["filtered-product-stress"]["answer"])


def test_realizability_of_tau():
    u = rng.normal(size=512)
    tau, ub = tau_block(u, 8)
    assert np.all(tau >= -1e-14)
    assert tau.mean() > 0                                   # 평균은 0이 아니다(선택지 2 오답)
    ux = np.roll(ub, -1) - np.roll(ub, 1)
    assert np.any(tau * ux > 0) and np.any(tau * ux < 0)    # 부호가 −ū_x와 늘 같지 않다(선택지 1 오답)
    assert correct("realizability-of-tau") == [0]


def test_eddy_viscosity_dissipation():
    ub = rng.normal(size=64)
    ux = (np.roll(ub, -1) - np.roll(ub, 1)) / 2
    nut = np.abs(rng.normal(size=64))
    assert np.sum(-nut * ux * ux) <= 0
    assert correct("eddy-viscosity-dissipation") == [0]


def test_galilean_invariance():
    u = rng.normal(size=256)
    V = 2.5
    t0, ub0 = tau_block(u, 8)
    t1, ub1 = tau_block(u + V, 8)
    assert np.allclose(t0, t1)                              # 참 τ는 불변
    ux0 = np.roll(ub0, -1) - np.roll(ub0, 1)
    ux1 = np.roll(ub1, -1) - np.roll(ub1, 1)
    assert np.allclose(ux0, ux1)                            # 기울기도 불변
    assert not np.allclose(ub0 * ux0, ub1 * ux1)            # ū ū_x는 바뀐다
    assert correct("galilean-invariant-features") == [0]


def test_choice_keys():
    assert correct("a-priori-not-enough") == [0]
    assert correct("inviscid-limit-check") == [0, 1]
