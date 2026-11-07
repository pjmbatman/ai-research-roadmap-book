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


m_ = load()


def run(step, grad, x0, lr, wd, steps):
    x = np.array(x0, dtype=float)
    m, v = np.zeros_like(x), np.zeros_like(x)
    for t in range(1, steps + 1):
        x, m, v = step(x, grad(x), m, v, t, lr, wd)
    return x


def test_no_decay_is_same():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(10, 3))
    x1 = x2 = np.ones(3)
    m1 = v1 = m2 = v2 = np.zeros(3)
    for t, g in enumerate(G, start=1):
        x1, m1, v1 = m_.adam_l2_step(x1, g, m1, v1, t, 0.01, 0.0)
        x2, m2, v2 = m_.adamw_step(x2, g, m2, v2, t, 0.01, 0.0)
    assert np.allclose(x1, x2), "wd = 0이면 두 방식은 같아야 합니다"


def test_adamw_decays_multiplicatively():
    x0, lr, wd = np.array([0.001, 1.0, 1000.0]), 0.01, 0.1
    x = run(m_.adamw_step, lambda x: np.zeros(3), x0, lr, wd, 100)
    assert np.allclose(x, x0 * (1 - lr * wd) ** 100), "데이터 gradient가 0이면 AdamW는 매 step (1 - lr*wd)배가 됩니다"


def test_adam_l2_decay_is_normalized():
    x0, lr, wd = np.array([10.0, 1000.0]), 0.01, 0.1
    x = run(m_.adam_l2_step, lambda x: np.zeros(2), x0, lr, wd, 100)
    assert np.allclose(x0 - x, 100 * lr, rtol=0.05), "Adam + L2에서는 감쇠 항도 정규화되어 크기와 무관하게 step마다 약 lr씩 줄어듭니다"


def test_fixed_points_differ():
    a, lam = np.array([0.01, 1.0, 100.0]), 1.0
    grad = lambda x: a * (x - 1.0)
    x_l2 = run(m_.adam_l2_step, grad, np.zeros(3), 0.01, lam, 20000)
    x_w = run(m_.adamw_step, grad, np.zeros(3), 0.01, lam, 20000)
    assert np.allclose(x_l2, a / (a + lam), atol=1e-3), "Adam + L2는 L2 정규화 문제의 해 a/(a + lam)로 가야 합니다"
    assert np.ptp(x_w) < 1e-3, "AdamW는 세 좌표를 같은 값으로 보내야 합니다"
    assert np.ptp(x_l2) > 0.9, "Adam + L2의 감쇠는 곡률에 따라 좌표마다 크게 다릅니다"
