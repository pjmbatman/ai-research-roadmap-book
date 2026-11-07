import importlib.util
import math
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


nt_xent = load().nt_xent
rng = np.random.default_rng(0)


def naive(z1, z2, tau):                               # 정의를 그대로 옮긴 반복문
    z = [v / np.linalg.norm(v) for v in list(z1) + list(z2)]
    N, total = len(z1), 0.0
    for a in range(2 * N):
        p = a + N if a < N else a - N
        denom = sum(math.exp(z[a] @ z[b] / tau) for b in range(2 * N) if b != a)
        total += -(z[a] @ z[p]) / tau + math.log(denom)
    return total / (2 * N)


def test_matches_definition():
    z1, z2 = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    assert np.isclose(nt_xent(z1, z2, 0.5), naive(z1, z2, 0.5)), "정의대로 계산한 값과 다릅니다"


def test_collapse_is_log_2n_minus_1():
    z = np.ones((8, 3))
    assert np.isclose(nt_xent(z, z, 0.2), np.log(15)), "모든 embedding이 같으면 log(2N-1) = log 15여야 합니다"


def test_perfect_pairs_approach_zero():
    z = np.eye(5)                                     # 짝끼리 같고, 서로 다른 표본은 직교
    big, small = nt_xent(z, z, 1.0), nt_xent(z, z, 0.05)
    assert small < 1e-6 < big, f"τ가 작을수록 손실이 0에 가까워져야 합니다: τ=1 {big:.4f}, τ=0.05 {small:.2e}"


def test_scale_invariance_and_symmetry():
    z1, z2 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    scaled = z1 * rng.uniform(0.1, 10, size=(5, 1))
    assert np.isclose(nt_xent(z1, z2, 0.3), nt_xent(scaled, z2, 0.3)), "행마다 양수를 곱해도 값이 같아야 합니다(정규화)"
    assert np.isclose(nt_xent(z1, z2, 0.3), nt_xent(z2, z1, 0.3)), "z1과 z2를 바꿔도 값이 같아야 합니다"


def test_small_tau_is_finite():
    z1 = rng.normal(size=(10, 4))
    val = nt_xent(z1, z1 + 0.01 * rng.normal(size=(10, 4)), 0.01)
    assert np.isfinite(val), "τ가 작아도 inf·nan 없이 계산해야 합니다(log-sum-exp)"
