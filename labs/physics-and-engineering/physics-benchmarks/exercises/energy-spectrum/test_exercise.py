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


energy_spectrum = load().energy_spectrum


def test_parseval():
    u = np.random.default_rng(0).normal(size=128)
    k, E = energy_spectrum(u, 2 * np.pi)
    assert E.shape == (65,) and k.shape == (65,), f"shape은 (N//2 + 1,)이어야 합니다. 현재 {k.shape}, {E.shape}"
    assert np.isclose(E.sum(), np.mean(u**2) / 2), f"Parseval: sum(E) = {E.sum():.6f}, mean(u²)/2 = {np.mean(u**2) / 2:.6f}"


def test_single_mode():
    L, N, A = 3.0, 64, 1.7
    x = np.arange(N) * L / N
    k, E = energy_spectrum(A * np.sin(3 * 2 * np.pi * x / L), L)
    assert np.isclose(E[3], A**2 / 4), f"모드 3의 에너지는 A²/4 = {A**2 / 4:.4f}이어야 합니다. 현재 {E[3]:.4f}"
    assert np.allclose(np.delete(E, 3), 0, atol=1e-12), "다른 모드에는 에너지가 없어야 합니다"
    assert np.isclose(k[3], 2 * np.pi * 3 / L), "파수는 2π m / L입니다"


def test_constant_field():
    _, E = energy_spectrum(np.full(32, -0.8), 2 * np.pi)
    assert np.isclose(E[0], 0.32), f"상수 장의 E_0 = c²/2 = 0.32이어야 합니다. 현재 {E[0]:.4f}"
    assert np.allclose(E[1:], 0, atol=1e-14)


def test_nyquist_and_integer_wavenumbers():
    N = 16
    u = np.cos(np.pi * np.arange(N))              # +1, -1, +1, ... : Nyquist 모드만
    k, E = energy_spectrum(u, 2 * np.pi)
    assert np.allclose(k, np.arange(N // 2 + 1)), "L = 2π이면 파수는 0, 1, ..., N/2입니다"
    assert np.isclose(E[-1], 0.5) and np.isclose(E.sum(), 0.5), f"Nyquist 모드의 에너지는 1/2이어야 합니다. 현재 {E[-1]:.4f}"
