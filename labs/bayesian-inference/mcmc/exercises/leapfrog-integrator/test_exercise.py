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


m = load()
log_p = lambda th: -0.5 * th @ th                     # 조화 진동자, omega = 1
grad = lambda th: -th


def energy_error(eps, L):
    th0, r0 = np.array([1.0]), np.array([0.5])
    th, r = m.leapfrog(th0, r0, grad, eps, L)
    return abs(m.hamiltonian(th, r, log_p) - m.hamiltonian(th0, r0, log_p))


def test_hamiltonian():
    assert m.hamiltonian(np.array([1.0, 2.0]), np.array([3.0, 0.0]), log_p) == pytest.approx(2.5 + 4.5)


def test_single_step_formula():
    th, r = m.leapfrog(np.array([1.0]), np.array([0.0]), grad, 0.1, 1)
    # r: 0 - 0.05 * 1 = -0.05,  th: 1 - 0.005 = 0.995,  r: -0.05 - 0.05 * 0.995
    assert th[0] == pytest.approx(0.995)
    assert r[0] == pytest.approx(-0.05 - 0.05 * 0.995)


def test_energy_error_is_second_order():
    e1, e2 = energy_error(0.2, 15), energy_error(0.1, 30)   # 같은 적분 시간 3
    assert e1 < 0.02, "보폭이 작으면 에너지가 거의 보존되어야 합니다"
    assert 2.5 < e1 / e2 < 6.0, "보폭을 절반으로 줄이면 에너지 오차가 약 1/4이 되어야 합니다"


def test_reversible():
    th0, r0 = np.array([0.3, -1.2]), np.array([0.7, 0.4])
    th1, r1 = m.leapfrog(th0, r0, grad, 0.25, 12)
    th2, r2 = m.leapfrog(th1, -r1, grad, 0.25, 12)
    assert np.allclose(th2, th0) and np.allclose(-r2, r0), "운동량을 뒤집어 다시 적분하면 출발점으로 돌아와야 합니다"


def test_does_not_modify_inputs():
    th0, r0 = np.array([1.0, 2.0]), np.array([0.5, -0.5])
    m.leapfrog(th0, r0, grad, 0.1, 5)
    assert np.array_equal(th0, [1.0, 2.0]) and np.array_equal(r0, [0.5, -0.5]), "입력 배열을 제자리에서 바꾸면 안 됩니다"


def test_unstable_step_diverges():
    assert energy_error(2.5, 30) > 1e3, "eps * omega > 2이면 leapfrog가 발산해야 합니다"
