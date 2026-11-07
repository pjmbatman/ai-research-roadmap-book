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


mod = load()
ftcs_step, max_stable_dt = mod.ftcs_step, mod.max_stable_dt


def test_hand_computed_step():
    u = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    got = ftcs_step(u, 0.25)
    # 내부점: 1 + 0.25(3 - 2 + 0) = 1.25, 3 + 0.25(2 - 6 + 1) = 2.25, 2 + 0.25(5 - 4 + 3) = 3.0
    assert np.allclose(got, [0.0, 1.25, 2.25, 3.0, 5.0]), f"한 step 결과가 다릅니다: {got}"


def test_boundaries_fixed_and_no_mutation():
    u = np.linspace(1.0, 2.0, 11) ** 2
    before = u.copy()
    got = ftcs_step(u, 0.4)
    assert np.allclose(u, before), "입력 u를 제자리에서 바꾸면 안 됩니다"
    assert got[0] == u[0] and got[-1] == u[-1], "양 끝 값(Dirichlet 경계)은 그대로여야 합니다"
    assert got is not u, "새 배열을 반환해야 합니다"


@pytest.mark.parametrize("k,r", [(1, 0.25), (3, 0.4), (7, 0.5)])
def test_amplification_factor(k, r):
    N = 20
    x = np.linspace(0, 1, N + 1)
    u = np.sin(k * np.pi * x)
    g = 1 - 4 * r * np.sin(k * np.pi / (2 * N)) ** 2
    assert np.allclose(ftcs_step(u, r), g * u, atol=1e-12), "격자 모드는 한 step에 증폭 인자 g배가 되어야 합니다"


def test_max_stable_dt():
    assert np.isclose(max_stable_dt(0.01, 1e-4), 0.5), "Δx = 0.01 m, α = 1e-4 m²/s이면 Δt ≤ 0.5 s"
    assert np.isclose(max_stable_dt(0.1, 2.0), 0.0025)


def test_stability_boundary():
    N, alpha = 20, 1.0
    dx = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    u0 = np.sin(np.pi * x) + 1e-6 * np.sin((N - 1) * np.pi * x)   # 가장 빠른 모드를 조금 섞는다
    for factor, grows in [(1.0, False), (1.1, True)]:
        r = alpha * factor * max_stable_dt(dx, alpha) / dx**2
        u = u0.copy()
        for _ in range(400):
            u = ftcs_step(u, r)
        assert (np.max(np.abs(u)) > 1.0) == grows, f"Δt = {factor} × 한계에서 {'커져야' if grows else '커지지 않아야'} 합니다"
