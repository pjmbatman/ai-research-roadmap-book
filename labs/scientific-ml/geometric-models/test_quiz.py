"""24.8 퀴즈(content/24-scientific-ml/08-geometric-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/24-scientific-ml/08-geometric-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
IU = np.triu_indices(4, 1)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_internal_dof():
    x = np.random.default_rng(0).normal(size=(4, 3))
    dists = lambda flat: np.linalg.norm(flat.reshape(4, 3)[:, None] - flat.reshape(4, 3)[None], axis=-1)[IU]
    h = 1e-6
    J = np.column_stack([(dists(x.ravel() + h * e) - dists(x.ravel() - h * e)) / (2 * h) for e in np.eye(12)])
    assert np.linalg.matrix_rank(J, tol=1e-6) == Q["internal-dof"]["answer"] == 3 * 4 - 6


def test_force_from_invariant_energy_is_equivariant():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 3))
    E = lambda x: np.sum(np.exp(-np.linalg.norm(x[:, None] - x[None], axis=-1)[IU]))
    def force(x, h=1e-6):
        return -np.array([[(E(x + h * np.eye(12)[3 * i + k].reshape(4, 3)) - E(x - h * np.eye(12)[3 * i + k].reshape(4, 3))) / (2 * h)
                           for k in range(3)] for i in range(4)])
    R = rot_z(0.7) @ rot_x(1.1)
    assert np.allclose(force(x @ R.T), force(x) @ R.T, atol=1e-6)
    assert correct("force-target") == [0]


def test_egnn_message_inputs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 3))
    W = rng.normal(size=(3, 5))
    msg_vec = lambda x: np.tanh((x[:, None] - x[None]) @ W)           # 상대 벡터를 MLP에 넣은 메시지
    R, t = rot_z(0.9) @ rot_x(0.4), rng.normal(size=3)
    assert not np.allclose(msg_vec(x @ R.T), msg_vec(x))              # 회전에 불변 아님
    assert np.allclose(msg_vec(x + t), msg_vec(x))                    # 평행이동에는 불변
    assert correct("egnn-message-inputs") == [0]


def test_field_breaks_symmetry():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 3))
    def field_term(x):
        d = (x[:, None] - x[None])[IU]
        return np.sum(d[:, 2] ** 2 / np.sum(d**2, axis=1))
    assert np.isclose(field_term(x @ rot_z(1.3).T + rng.normal(size=3)), field_term(x))   # z축 회전·이동: 불변
    assert not np.isclose(field_term(x @ rot_x(1.0).T), field_term(x))                     # 다른 축 회전: 바뀜
    assert correct("field-breaks-symmetry") == [0]


def test_choice_questions():
    assert correct("augmentation-vs-equivariance") == [0, 1]
    assert correct("rotated-test-claim") == [0]
