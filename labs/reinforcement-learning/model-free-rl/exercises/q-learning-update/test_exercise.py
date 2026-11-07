import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


q_learning_update = load().q_learning_update


def test_single_update_by_hand():
    Q = np.zeros((2, 2))
    Q[0, 1] = 2.0
    Q[1] = [3.0, -1.0]                       # max_a' Q(s', a') = 3
    new = q_learning_update(Q, s=0, a=1, r=1.0, s_next=1, done=False, alpha=0.5, gamma=0.9)
    assert np.isclose(new[0, 1], 2.85), "목표 1 + 0.9 * 3 = 3.7, 갱신 2 + 0.5 * (3.7 - 2) = 2.85입니다"


def test_only_one_entry_changes():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 3))
    new = q_learning_update(Q, s=2, a=0, r=0.5, s_next=4, done=False, alpha=0.1, gamma=0.99)
    mask = np.ones_like(Q, dtype=bool)
    mask[2, 0] = False
    assert np.array_equal(new[mask], Q[mask]), "Q[s, a] 말고 다른 원소는 바뀌면 안 됩니다"


def test_terminal_has_no_bootstrap():
    Q = np.full((2, 2), 10.0)                # 다음 상태의 값이 커도
    new = q_learning_update(Q, s=0, a=0, r=1.0, s_next=1, done=True, alpha=1.0, gamma=0.9)
    assert np.isclose(new[0, 0], 1.0), "종료(done=True)이면 목표는 r뿐입니다"


def test_uses_max_not_mean():
    Q = np.zeros((2, 3))
    Q[1] = [0.0, 4.0, -8.0]
    new = q_learning_update(Q, s=0, a=2, r=0.0, s_next=1, done=False, alpha=1.0, gamma=0.5)
    assert np.isclose(new[0, 2], 2.0), "Q-learning 목표는 다음 상태의 max_a' Q입니다(평균이나 실제 행동이 아님)"


def test_converges_to_q_star_on_chain():
    # 결정적 사슬 0 -> 1 -> 2(종료). 행동 0 = 전진(보상 0, 마지막에 1), 행동 1 = 제자리(보상 0).
    gamma = 0.9
    Q = np.zeros((3, 2))
    rng = np.random.default_rng(1)
    for _ in range(3000):
        s, a = int(rng.integers(2)), int(rng.integers(2))
        if a == 0:
            s2, r, done = s + 1, float(s == 1), s == 1
        else:
            s2, r, done = s, 0.0, False
        Q = q_learning_update(Q, s, a, r, s2, done, alpha=0.2, gamma=gamma)
    expected = np.array([[gamma, gamma**2], [1.0, gamma]])
    assert np.allclose(Q[:2], expected, atol=1e-3), "여러 번 갱신하면 Q*로 수렴해야 합니다"
