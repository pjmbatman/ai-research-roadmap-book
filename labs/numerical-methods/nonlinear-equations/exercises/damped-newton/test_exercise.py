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


damped_newton = load().damped_newton


def F_circle(v):                                  # 원 x^2 + y^2 = 4와 곡선 y = 1 - e^x의 교점
    x, y = v
    return np.array([x**2 + y**2 - 4, np.exp(x) + y - 1])


def J_circle(v):
    x, y = v
    return np.array([[2 * x, 2 * y], [np.exp(x), 1.0]])


def F_atan(v):                                    # 근은 원점. 순수 Newton은 |x0| > 1.39 근처에서 발산
    return np.arctan(v)


def J_atan(v):
    return np.diag(1 / (1 + v**2))


def pure_newton(F, J, x, n):
    for _ in range(n):
        x = x + np.linalg.solve(J(x), -F(x))
    return x


def test_converges_near_root_quickly():
    x, k = damped_newton(F_circle, J_circle, np.array([1.0, -1.0]))
    assert np.linalg.norm(F_circle(x)) <= 1e-10
    assert np.allclose(x, [1.0041687385, -1.7296372870], atol=1e-9)
    assert k <= 7, f"근 근처에서 Newton은 몇 번이면 수렴합니다(2차 수렴). 현재 {k}번"


def test_other_root_from_other_start():
    x, _ = damped_newton(F_circle, J_circle, np.array([-2.0, 1.0]))
    assert np.linalg.norm(F_circle(x)) <= 1e-10 and x[0] < 0, "왼쪽 위의 다른 교점으로 가야 합니다"


def test_damping_rescues_divergent_start():
    x0 = np.array([1.5, -2.0])
    with np.errstate(all="ignore"):
        bad = pure_newton(F_atan, J_atan, x0, 6)
    assert not np.all(np.abs(bad) < 1), "이 초기값에서 순수 Newton은 발산합니다(전제 확인)"
    x, k = damped_newton(F_atan, J_atan, x0)
    assert np.linalg.norm(x) <= 1e-9, f"damped Newton은 원점으로 수렴해야 합니다. 현재 {x}"
    assert k < 50


def test_returns_immediately_at_root():
    x, k = damped_newton(F_atan, J_atan, np.zeros(3))
    assert k == 0 and np.allclose(x, 0)


def test_far_start_in_three_dimensions():
    x, _ = damped_newton(F_atan, J_atan, np.array([3.0, -4.0, 2.5]))
    assert np.linalg.norm(x) <= 1e-9, f"3차원에서도 원점으로 수렴해야 합니다. 현재 {x}"
