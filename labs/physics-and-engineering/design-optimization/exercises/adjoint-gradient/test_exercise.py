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


adjoint_gradient = load().adjoint_gradient


def reference_J(k, f, c, h):
    """요소 행렬을 하나씩 더해 만든 K로 J를 계산한다(테스트용 기준)."""
    n = len(k)
    Kf = np.zeros((n + 1, n + 1))
    for e in range(n):
        Kf[e:e + 2, e:e + 2] += k[e] / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return c @ np.linalg.solve(Kf[1:, 1:], f)


def problem(n=12, seed=0):
    rng = np.random.default_rng(seed)
    h = 1.0 / n
    k = rng.uniform(0.5, 2.0, size=n)
    f = rng.uniform(0.0, 1.0, size=n) * h
    return k, f, h, rng


def test_value_matches_reference():
    k, f, h, rng = problem()
    c = rng.normal(size=len(k))
    J, _ = adjoint_gradient(k, f, c, h)
    assert np.isclose(J, reference_J(k, f, c, h)), "J = c^T T가 요소 조립으로 만든 기준과 달라요. K의 구성을 확인하세요"


@pytest.mark.parametrize("kind", ["last", "random", "mean"])
def test_gradient_matches_finite_difference(kind):
    k, f, h, rng = problem(seed=1)
    n = len(k)
    c = {"last": np.eye(n)[-1], "random": rng.normal(size=n), "mean": np.full(n, 1.0 / n)}[kind]
    _, g = adjoint_gradient(k, f, c, h)
    eps = 1e-6
    g_fd = np.array([(reference_J(k + eps * np.eye(n)[e], f, c, h) - reference_J(k - eps * np.eye(n)[e], f, c, h))
                     / (2 * eps) for e in range(n)])
    assert np.allclose(g, g_fd, rtol=1e-5, atol=1e-9), f"adjoint 기울기와 중심 유한차분이 다릅니다.\n{g}\n{g_fd}"


def test_compliance_gradient_nonpositive():
    k, f, h, _ = problem(seed=2)
    _, g = adjoint_gradient(k, f, f, h)
    assert np.all(g <= 1e-14), "c = f(열 compliance)이면 모든 요소에서 dJ/dk <= 0이어야 합니다"


def test_solve_count(monkeypatch):
    calls = {"n": 0}
    real = np.linalg.solve

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", counting)
    k, f, h, rng = problem(n=200, seed=3)
    adjoint_gradient(k, f, rng.normal(size=200), h)
    assert calls["n"] <= 3, f"설계 변수 200개에 np.linalg.solve를 {calls['n']}회 불렀습니다. adjoint는 2회면 됩니다"
