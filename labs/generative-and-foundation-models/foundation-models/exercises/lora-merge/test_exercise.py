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
lora_forward, merge, n_trainable = mod.lora_forward, mod.merge, mod.n_trainable
rng = np.random.default_rng(0)
d_in, d_out, r = 12, 7, 3
W0, A, B = rng.normal(size=(d_out, d_in)), rng.normal(size=(r, d_in)), rng.normal(size=(d_out, r))
X = rng.normal(size=(5, d_in))


def test_forward_shape_and_value():
    Y = lora_forward(X, W0, A, B, alpha=6.0)
    assert np.shape(Y) == (5, d_out), f"출력 shape은 (N, d_out) = (5, {d_out})이어야 합니다. 현재 {np.shape(Y)}"
    expected = X @ W0.T + (6.0 / r) * X @ A.T @ B.T
    assert np.allclose(Y, expected), "y = W0 x + (alpha/r) B A x를 확인하세요 (r은 A의 행 수)"


def test_merge_matches_forward():
    W = merge(W0, A, B, alpha=2.0)
    assert np.shape(W) == (d_out, d_in), "병합한 가중치의 shape은 (d_out, d_in)이어야 합니다"
    assert np.allclose(X @ W.T, lora_forward(X, W0, A, B, alpha=2.0)), "병합한 가중치의 출력은 병합하지 않은 두 경로 출력과 같아야 합니다"


def test_zero_B_is_pretrained():
    Y = lora_forward(X, W0, A, np.zeros((d_out, r)), alpha=8.0)
    assert np.allclose(Y, X @ W0.T), "B = 0이면 사전학습 모델과 같은 출력이어야 합니다"


def test_update_is_low_rank_and_W0_untouched():
    W0_copy = W0.copy()
    W = merge(W0, A, B, alpha=1.0)
    assert np.array_equal(W0, W0_copy), "merge가 W0를 제자리에서 바꾸면 안 됩니다"
    assert np.linalg.matrix_rank(W - W0) <= r, "갱신 W - W0의 rank는 r 이하여야 합니다"


def test_trainable_count():
    assert n_trainable(1024, 1024, 8) == 16384, "r (d_in + d_out) = 8 × 2048 = 16384"
    assert n_trainable(64, 256, 4) == 4 * (64 + 256)
