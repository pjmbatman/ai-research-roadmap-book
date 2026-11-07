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


bfgs_update = load().bfgs_update


def random_pair(rng, n):
    A = rng.normal(size=(n, n))
    A = A @ A.T + n * np.eye(n)                             # 대칭 양의 정부호 Hessian
    s = rng.normal(size=n)
    return A, s, A @ s                                      # 이차함수에서 y = A s


def test_secant_condition():
    rng = np.random.default_rng(0)
    _, s, y = random_pair(rng, 4)
    Hn = bfgs_update(np.eye(4), s, y)
    assert np.allclose(Hn @ y, s), "갱신한 역Hessian 근사는 secant 조건 H_new y = s를 만족해야 합니다"


def test_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    H = np.eye(5)
    for _ in range(10):
        _, s, y = random_pair(rng, 5)
        H = bfgs_update(H, s, y)
        assert np.allclose(H, H.T), "갱신 결과는 대칭이어야 합니다"
        assert np.linalg.eigvalsh(H).min() > 0, "s^T y > 0이면 양의 정부호가 유지되어야 합니다"


def test_input_not_modified():
    rng = np.random.default_rng(2)
    _, s, y = random_pair(rng, 3)
    H = np.eye(3)
    bfgs_update(H, s, y)
    assert np.allclose(H, np.eye(3)), "입력 행렬 Hinv를 제자리에서 바꾸지 않습니다"


def test_recovers_inverse_hessian_on_quadratic():
    rng = np.random.default_rng(3)
    n = 4
    A, _, _ = random_pair(rng, n)
    b = rng.normal(size=n)
    x, H = np.zeros(n), np.eye(n)
    for _ in range(n):                                      # exact line search를 쓰는 BFGS
        g = A @ x - b
        d = -H @ g
        t = -(g @ d) / (d @ A @ d)
        s = t * d
        y = A @ s
        x, H = x + s, bfgs_update(H, s, y)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8), "n step 뒤 최솟점에 도달해야 합니다"
    assert np.allclose(H, np.linalg.inv(A), atol=1e-8), "n step 뒤 역Hessian 근사가 A^{-1}과 같아야 합니다"
