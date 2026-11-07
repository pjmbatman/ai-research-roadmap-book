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


pod_basis = load().pod_basis


def known_data(svals, M=20, N=30, seed=0):
    """평균을 뺀 행렬의 특이값이 svals인 스냅샷 행렬을 만든다."""
    rng = np.random.default_rng(seed)
    K = len(svals)
    U, _ = np.linalg.qr(rng.normal(size=(M, K)))
    U -= U.mean(axis=0)                                      # 열의 평균을 0으로(평균을 빼도 그대로 남도록)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(rng.normal(size=(N, K)))
    base = rng.normal(size=N)
    return base + U @ np.diag(svals) @ V.T


def test_shapes_and_orthonormal():
    X = np.random.default_rng(1).normal(size=(12, 40))
    mean, modes = pod_basis(X, 0.9)
    assert mean.shape == (40,), f"mean의 shape은 (N,)이어야 합니다. 현재 {np.shape(mean)}"
    assert modes.ndim == 2 and modes.shape[1] == 40, f"modes의 shape은 (r, N)이어야 합니다. 현재 {np.shape(modes)}"
    assert np.allclose(modes @ modes.T, np.eye(len(modes)), atol=1e-10), "모드는 정규직교여야 합니다"
    assert np.allclose(mean, X.mean(0)), "mean은 스냅샷의 평균이어야 합니다"


def test_selects_smallest_r():
    s = np.array([10.0, 5.0, 2.0, 1.0, 0.1])
    X = known_data(s)
    ratio = np.cumsum(s**2) / np.sum(s**2)                   # [0.76917, 0.96146, 0.99223, 0.99992, 1]
    for energy, r in [(0.5, 1), (0.95, 2), (0.99, 3), (0.9999, 4), (0.99995, 5)]:
        _, modes = pod_basis(X, energy)
        assert len(modes) == r, f"energy={energy}: 누적 비율 {np.round(ratio, 5)}에서 r={r}이어야 합니다. 현재 {len(modes)}"


def test_reconstruction_error_equals_discarded_energy():
    s = np.array([8.0, 3.0, 1.0, 0.5])
    X = known_data(s, seed=2)
    mean, modes = pod_basis(X, 0.95)
    Xc = X - mean
    recon = Xc @ modes.T @ modes
    err2 = np.sum((Xc - recon) ** 2)
    discarded = np.sum(s[len(modes):] ** 2)
    assert np.isclose(err2, discarded), f"복원 오차 제곱합 {err2:.4f}, 버린 에너지 {discarded:.4f}"


def test_modes_preserve_mass():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 64))
    X += (2.0 - X.sum(1, keepdims=True)) / 64                 # 모든 스냅샷의 성분 합을 2로
    mean, modes = pod_basis(X, 0.99)
    assert np.allclose(modes.sum(1), 0, atol=1e-10), "같은 질량의 스냅샷에서 만든 모드는 성분 합이 0이어야 합니다"
    pred = mean + rng.normal(size=len(modes)) @ modes
    assert np.isclose(pred.sum(), 2.0), "평균 + 모드의 선형 결합은 질량을 보존해야 합니다"
