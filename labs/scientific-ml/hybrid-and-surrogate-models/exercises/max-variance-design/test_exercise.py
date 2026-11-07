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


next_point = load().next_point


def reference_var(x_tr, x_cand, ell, sf, sn):
    k = lambda a, b: sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    Ky = k(x_tr, x_tr) + sn**2 * np.eye(len(x_tr))
    ks = k(x_tr, x_cand)
    return sf**2 - np.einsum("im,im->m", ks, np.linalg.solve(Ky, ks))


def test_shape_and_formula():
    rng = np.random.default_rng(0)
    x_tr, x_cand = rng.uniform(0, 1, 6), np.linspace(0, 1, 51)
    x_next, var = next_point(x_tr, x_cand, ell=0.2, sf=1.5, sn=0.1)
    assert np.shape(var) == (51,), f"var의 shape은 (M,) = (51,)이어야 합니다. 현재 {np.shape(var)}"
    assert np.allclose(var, reference_var(x_tr, x_cand, 0.2, 1.5, 0.1)), "사후분산 식이 다릅니다"
    assert np.isclose(x_next, x_cand[np.argmax(var)]), "x_next는 var가 가장 큰 후보여야 합니다"


def test_picks_middle_of_gap():
    x_next, _ = next_point(np.array([0.0, 1.0]), np.linspace(0, 1, 101), ell=0.2)
    assert np.isclose(x_next, 0.5), f"두 점 0과 1 사이에서는 가운데 0.5가 가장 불확실합니다. 현재 {x_next}"


def test_variance_limits():
    x_tr = np.array([0.0, 0.3])
    _, var = next_point(x_tr, np.array([0.0, 0.3, 10.0]), ell=0.1, sf=2.0, sn=1e-3)
    assert np.all(var[:2] < 1e-3), "이미 실행한 점에서는 사후분산이 거의 0이어야 합니다"
    assert np.isclose(var[2], 4.0), "데이터에서 아주 먼 점의 사후분산은 사전분산 sf^2 = 4여야 합니다"


def test_variance_nonnegative_and_bounded():
    rng = np.random.default_rng(1)
    _, var = next_point(rng.uniform(0, 1, 10), np.linspace(-0.5, 1.5, 81), ell=0.15, sf=1.0, sn=0.05)
    assert np.all(var > -1e-12) and np.all(var <= 1.0 + 1e-12), "사후분산은 0과 사전분산 sf^2 사이여야 합니다"
