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


mod = load()
conservation_laws, max_violation = mod.conservation_laws, mod.max_violation

CHAIN = np.array([[-1.0, 0.0],      # A → B → C
                  [1.0, -1.0],
                  [0.0, 1.0]])
REVERSIBLE = np.array([[-1.0, 1.0],  # A + B ⇌ C
                       [-1.0, 1.0],
                       [1.0, -1.0]])


def simulate(N, rates, y0, h=1e-3, steps=2000):
    """explicit Euler로 반응 네트워크를 적분한다. 선형 보존량은 Euler에서도 정확히 보존된다."""
    y, out = np.array(y0, dtype=float), [np.array(y0, dtype=float)]
    for _ in range(steps):
        y = y + h * N @ rates(y)
        out.append(y.copy())
    return np.array(out)                                  # (steps + 1, S)


def test_chain_total_mass():
    W = conservation_laws(CHAIN)
    assert np.shape(W) == (1, 3), f"A → B → C의 보존량은 1개(총량)여야 합니다. 현재 shape {np.shape(W)}"
    w = W[0] / W[0][0]
    assert np.allclose(w, [1, 1, 1]), "A → B → C에서 보존되는 것은 A + B + C입니다"


def test_reversible_two_laws():
    W = conservation_laws(REVERSIBLE)
    assert np.shape(W) == (2, 3), f"A + B ⇌ C는 rank 1이므로 보존량이 2개여야 합니다. 현재 {np.shape(W)}"
    assert np.allclose(W @ REVERSIBLE, 0, atol=1e-10), "W @ N이 0이어야 합니다"
    assert np.allclose(W @ W.T, np.eye(2), atol=1e-10), "W의 행은 정규직교여야 합니다"
    for v in ([1, 0, 1], [0, 1, 1]):                      # A + C, B + C는 보존량 공간 안에 있어야 한다
        v = np.array(v, dtype=float)
        assert np.allclose(W.T @ (W @ v), v), f"{v}가 보존량 공간에 들어 있지 않습니다"


def test_no_conservation():
    W = conservation_laws(np.eye(2))
    assert np.shape(W) == (0, 2), f"N이 full row rank이면 보존량이 없어 shape (0, 2)여야 합니다. 현재 {np.shape(W)}"


def test_violation_exact_and_leaky():
    k1, k2 = 2.0, 0.5
    rates = lambda y: np.array([k1 * y[0], k2 * y[1]])
    Y = simulate(CHAIN, rates, [1.0, 0.0, 0.0])
    W = conservation_laws(CHAIN)
    assert max_violation(Y, W) < 1e-10, "정확한 궤적에서는 보존량이 변하지 않아야 합니다"
    leaky = Y * np.exp(-0.1 * np.linspace(0, 2, len(Y)))[:, None]   # 시간에 따라 질량이 새는 모델
    v = max_violation(leaky, W)
    assert 0.1 < v < 0.2, f"질량이 약 18% 새는 궤적의 위반량(총량 기준 ≈ 0.105)을 잡아야 합니다. 현재 {v}"
