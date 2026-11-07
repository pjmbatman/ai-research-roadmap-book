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


spectral_conv1d = load().spectral_conv1d


def band_limited(N, rng_seed=0, C=2, K_max=6):
    """파수 K_max 이하의 매끄러운 함수 C개를 N점 격자에서 샘플링한다. (C, N)"""
    rng = np.random.default_rng(rng_seed)
    x = np.arange(N) / N
    k = np.arange(1, K_max + 1)
    a, b = rng.normal(size=(C, K_max)), rng.normal(size=(C, K_max))
    return a @ np.cos(2 * np.pi * np.outer(k, x)) + b @ np.sin(2 * np.pi * np.outer(k, x)) + 0.3


def random_weights(C_in, C_out, K, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(C_in, C_out, K)) + 1j * rng.normal(size=(C_in, C_out, K))


def test_shape_and_real():
    out = spectral_conv1d(band_limited(32), random_weights(2, 3, 8))
    assert np.shape(out) == (3, 32), f"shape은 (C_out, N) = (3, 32)여야 합니다. 현재 {np.shape(out)}"
    assert np.isrealobj(out), "출력은 실수 배열이어야 합니다(irfft를 쓰면 실수가 됩니다)"


def test_resolution_invariance():
    w = random_weights(2, 3, 8)
    coarse = spectral_conv1d(band_limited(32), w)                    # (3, 32)
    fine = spectral_conv1d(band_limited(128), w)                     # (3, 128)
    assert np.allclose(fine[:, ::4], coarse, atol=1e-10), \
        "같은 함수를 32점과 128점에서 샘플링한 결과가 겹치는 점에서 같아야 합니다. Fourier 계수를 N으로 정규화했는지 확인하세요"


def test_truncation():
    x = np.arange(64) / 64
    u = np.sin(2 * np.pi * 10 * x)[None, :]                          # 파수 10만 가진 함수
    out = spectral_conv1d(u, random_weights(1, 1, 8))                # K = 8 < 10
    assert np.allclose(out, 0, atol=1e-12), "파수 K 이상의 성분은 출력에서 사라져야 합니다"


def test_derivative_weights():
    N, K = 64, 16
    x = np.arange(N) / N
    w = (2j * np.pi * np.arange(K)).reshape(1, 1, K)                 # 모드 k에 2πik를 곱하면 미분
    out = spectral_conv1d(np.sin(2 * np.pi * x)[None, :], w)
    assert np.allclose(out[0], 2 * np.pi * np.cos(2 * np.pi * x), atol=1e-10), \
        "가중치 2πik는 미분 연산자여야 합니다. 모드 0부터 K-1까지 가중치를 곱했는지 확인하세요"
