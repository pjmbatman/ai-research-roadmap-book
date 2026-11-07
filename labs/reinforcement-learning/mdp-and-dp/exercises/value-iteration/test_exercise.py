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


value_iteration = load().value_iteration


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


def test_two_choices_by_hand():
    # 상태 0: 행동 0 = 보상 1을 받고 끝(흡수 상태 1로), 행동 1 = 보상 0.3을 받고 상태 0에 머문다.
    # gamma = 0.9이면 계속 머무는 이득 0.3 / (1 - 0.9) = 3 > 1이므로 머무는 것이 최적.
    P = np.zeros((2, 2, 2))
    P[0, 0, 1] = 1.0
    P[0, 1, 0] = 1.0
    P[1, :, 1] = 1.0
    R = np.array([[1.0, 0.3], [0.0, 0.0]])
    V, pi = value_iteration(P, R, 0.9)
    assert np.allclose(V, [3.0, 0.0], atol=1e-6), "V(0) = max(1, 0.3 / (1 - 0.9)) = 3이어야 합니다"
    assert int(pi[0]) == 1
    V, pi = value_iteration(P, R, 0.5)       # 0.3 / 0.5 = 0.6 < 1이므로 바로 끝내는 것이 최적
    assert np.allclose(V, [1.0, 0.0], atol=1e-6)
    assert int(pi[0]) == 0, "할인율이 바뀌면 최적 정책도 바뀝니다"


def test_grid_world_matches_text():
    P, R = grid_world()
    V, pi = value_iteration(P, R, 0.95)
    assert np.shape(V) == (16,) and np.shape(pi) == (16,), "V와 pi의 shape은 (S,)여야 합니다"
    assert np.isclose(V[12], 0.703, atol=5e-4), "본문의 시작 칸 V* = 0.703과 다릅니다"
    expected = [1, 1, 1, None, 0, 0, 3, None, 0, 0, 0, 2, 0, 0, 0, 3]   # 본문의 화살표 (종료 상태 제외)
    for s, a in enumerate(expected):
        if a is not None:
            assert int(pi[s]) == a, f"상태 {s}의 행동이 본문의 최적 정책과 다릅니다"


def test_bellman_optimality_residual():
    P, R = grid_world(slip=0.2)
    V, pi = value_iteration(P, R, 0.9)
    Q = R + 0.9 * P @ V                      # (S, A)
    assert np.abs(V - Q.max(axis=1)).max() < 1e-6, "V가 Bellman 최적 방정식을 만족하지 않습니다"
    assert np.allclose(Q[np.arange(16), pi], Q.max(axis=1)), "pi는 Q에 대해 탐욕적이어야 합니다"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_mdp_beats_every_deterministic_policy_sampled(seed):
    rng = np.random.default_rng(seed)
    S, A, gamma = 5, 3, 0.8
    P = rng.dirichlet(np.ones(S), size=(S, A))   # (S, A, S)
    R = rng.normal(size=(S, A))
    V, _ = value_iteration(P, R, gamma)
    for _ in range(20):
        pi = rng.integers(A, size=S)
        V_pi = np.linalg.solve(np.eye(S) - gamma * P[np.arange(S), pi], R[np.arange(S), pi])
        assert np.all(V >= V_pi - 1e-6), "V*는 어떤 정책의 가치보다도 작지 않아야 합니다"
