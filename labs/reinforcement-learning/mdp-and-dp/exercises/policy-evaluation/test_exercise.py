import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate_policy = load().evaluate_policy


def grid_world(slip=0.1):
    """본문의 4×4 미끄러지는 격자 세계. P (16, 4, 16), R (16, 4)."""
    H = W = 4
    goal, pit = 3, 7
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def move(s, d):
        r, c = divmod(s, W)
        r2, c2 = r + moves[d][0], c + moves[d][1]
        return r2 * W + c2 if 0 <= r2 < H and 0 <= c2 < W else s

    P = np.zeros((16, 4, 16))
    for s in range(16):
        for a in range(4):
            if s in (goal, pit):
                P[s, a, s] = 1.0
                continue
            for d, p in [(a, 1 - 2 * slip), ((a + 1) % 4, slip), ((a + 3) % 4, slip)]:
                P[s, a, move(s, d)] += p
    r_enter = np.zeros(16)
    r_enter[goal], r_enter[pit] = 1.0, -1.0
    R = P @ r_enter
    R[[goal, pit]] = 0.0
    return P, R


def test_single_state_geometric_series():
    P = np.ones((1, 1, 1))                   # 상태 하나, 행동 하나, 자기 자신으로
    R = np.array([[2.0]])
    V = evaluate_policy(P, R, np.ones((1, 1)), 0.9)
    assert np.shape(V) == (1,), f"V의 shape은 (S,) = (1,)이어야 합니다. 현재 {np.shape(V)}"
    assert np.isclose(V[0], 2.0 / (1 - 0.9)), "보상 2를 영원히 받으면 V = 2 / (1 - gamma) = 20입니다"


def test_two_state_by_hand():
    # 상태 0에서 행동 0: 보상 1, 상태 1로. 상태 1: 보상 0, 자기 자신에 머문다.
    P = np.zeros((2, 2, 2))
    P[0, 0, 1] = 1.0
    P[0, 1, 0] = 1.0                         # 행동 1: 보상 0, 상태 0에 머문다
    P[1, :, 1] = 1.0
    R = np.array([[1.0, 0.0], [0.0, 0.0]])
    V = evaluate_policy(P, R, np.array([[1.0, 0.0], [1.0, 0.0]]), 0.5)
    assert np.allclose(V, [1.0, 0.0])
    V = evaluate_policy(P, R, np.array([[0.5, 0.5], [1.0, 0.0]]), 0.5)
    # V0 = 0.5 * (1 + 0.5 * 0) + 0.5 * (0 + 0.5 * V0)  ->  V0 = 0.5 / 0.75
    assert np.allclose(V, [0.5 / 0.75, 0.0]), "확률적 정책은 행동별 기대값을 pi(a|s)로 가중평균합니다"


def test_bellman_residual_on_grid():
    P, R = grid_world()
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(4), size=16)  # (16, 4) 무작위 확률적 정책
    V = evaluate_policy(P, R, pi, 0.95)
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    assert np.abs(V - (r_pi + 0.95 * P_pi @ V)).max() < 1e-8, "V가 Bellman 기대 방정식을 만족하지 않습니다"


def test_uniform_policy_matches_text():
    P, R = grid_world()
    V = evaluate_policy(P, R, np.full((16, 4), 0.25), 0.95)
    assert np.isclose(V[2], 0.248, atol=5e-4), "본문의 무작위 정책 V(상태 2) = 0.248과 다릅니다"
    assert np.isclose(V[11], -0.531, atol=5e-4), "본문의 무작위 정책 V(상태 11) = -0.531과 다릅니다"


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99])
def test_gamma_zero_is_immediate_reward(gamma):
    P, R = grid_world()
    pi = np.zeros((16, 4))
    pi[:, 1] = 1.0                           # 항상 오른쪽
    V = evaluate_policy(P, R, pi, gamma)
    if gamma == 0.0:
        assert np.allclose(V, R[:, 1]), "gamma = 0이면 V는 즉시 보상과 같습니다"
    assert np.all(np.abs(V) <= 1.0 + 1e-9), "이 격자 세계의 이득은 -1과 1 사이입니다"
