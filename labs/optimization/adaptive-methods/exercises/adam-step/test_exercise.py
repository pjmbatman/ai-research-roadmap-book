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


adam_step = load().adam_step


def run(grads, lr=0.01):
    x, m, v = np.zeros(len(grads[0])), np.zeros(len(grads[0])), np.zeros(len(grads[0]))
    xs = []
    for t, g in enumerate(grads, start=1):
        x, m, v = adam_step(x, np.asarray(g, dtype=float), m, v, t, lr)
        xs.append(x.copy())
    return np.array(xs)


def test_first_step_is_lr_per_coordinate():
    x, m, v = adam_step(np.zeros(3), np.array([0.02, -50.0, 3.0]), np.zeros(3), np.zeros(3), 1, 0.001)
    assert np.allclose(np.abs(x), 0.001, rtol=1e-4), "bias correction 후 첫 step은 좌표마다 lr만큼 움직입니다"
    assert np.allclose(np.sign(x), [-1, 1, -1]), "gradient의 반대 방향으로 움직여야 합니다"


def test_returns_uncorrected_moments():
    g = np.array([2.0, -1.0])
    _, m, v = adam_step(np.zeros(2), g, np.zeros(2), np.zeros(2), 1, 0.1)
    assert np.allclose(m, 0.1 * g), "반환하는 m은 bias correction 전의 (1 - beta1) g입니다"
    assert np.allclose(v, 0.001 * g**2), "반환하는 v는 bias correction 전의 (1 - beta2) g^2입니다"


def test_scale_invariance():
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(20, 3))
    a = run(grads)
    b = run(grads * 1000.0)
    assert np.allclose(a, b, atol=1e-6), "gradient에 상수를 곱해도 Adam의 갱신은 (eps를 빼면) 같아야 합니다"


def test_matches_reference_sequence():
    grads = [[1.0, -2.0], [0.5, -1.0], [-0.2, 0.3]]
    xs = run(grads, lr=0.1)
    m, v, x = np.zeros(2), np.zeros(2), np.zeros(2)
    for t, g in enumerate(np.array(grads), start=1):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x = x - 0.1 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    assert np.allclose(xs[-1], x), "세 step 뒤의 값이 식과 다릅니다"


def test_converges_on_quadratic():
    lam = np.array([1.0, 100.0])                       # f(x) = sum lam (x - 1)^2 / 2
    x, m, v = np.zeros(2), np.zeros(2), np.zeros(2)
    for t in range(1, 3001):
        x, m, v = adam_step(x, lam * (x - 1.0), m, v, t, 0.01)
    assert np.allclose(x, 1.0, atol=0.02), "이차함수의 최솟점 (1, 1) 근처로 가야 합니다"
