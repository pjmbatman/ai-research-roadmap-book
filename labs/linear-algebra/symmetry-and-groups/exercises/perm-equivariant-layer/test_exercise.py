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


equivariant_layer = load().equivariant_layer


def params(rng, d_in=4, d_out=3):
    return rng.normal(size=(d_in, d_out)), rng.normal(size=(d_in, d_out)), rng.normal(size=d_out)


def reference(H, Ws, Wp, b):
    out = np.zeros(H.shape[:2] + (Ws.shape[1],))
    for i in range(H.shape[0]):
        mean = H[i].mean(axis=0)
        for n in range(H.shape[1]):
            out[i, n] = H[i, n] @ Ws + mean @ Wp + b
    return out


def test_matches_formula():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(2, 5, 4))                    # B = 2, N = 5, D_in = 4: 축 크기가 모두 다르다
    Ws, Wp, b = params(rng)
    out = equivariant_layer(H, Ws, Wp, b)
    assert np.shape(out) == (2, 5, 3), f"출력 shape은 (B, N, D_out) = (2, 5, 3)이어야 합니다. 현재 {np.shape(out)}"
    assert np.allclose(out, reference(H, Ws, Wp, b)), "식 H W_self + mean_n(H) W_pool + b와 다릅니다"


@pytest.mark.parametrize("seed", range(3))
def test_permutation_equivariance(seed):
    rng = np.random.default_rng(10 + seed)
    H = rng.normal(size=(3, 7, 4))
    Ws, Wp, b = params(rng)
    perms = [rng.permutation(7) for _ in range(3)]    # batch 항목마다 다른 순서
    H_perm = np.stack([H[i, p] for i, p in enumerate(perms)])
    out, out_perm = equivariant_layer(H, Ws, Wp, b), equivariant_layer(H_perm, Ws, Wp, b)
    for i, p in enumerate(perms):
        assert np.allclose(out_perm[i], out[i, p]), "원소 순서를 바꾸면 출력의 행도 같은 순서로 바뀌어야 합니다"


def test_batch_items_independent():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(4, 5, 4))
    Ws, Wp, b = params(rng)
    full = equivariant_layer(H, Ws, Wp, b)
    single = equivariant_layer(H[2:3], Ws, Wp, b)
    assert np.allclose(full[2:3], single), "평균은 batch 항목마다 따로 내야 합니다(batch 축 평균 금지)"


def test_different_set_sizes():
    rng = np.random.default_rng(2)
    Ws, Wp, b = params(rng)
    for N in [1, 3, 10]:
        H = rng.normal(size=(2, N, 4))
        assert np.allclose(equivariant_layer(H, Ws, Wp, b), reference(H, Ws, Wp, b)), \
            f"같은 가중치로 원소 수 N = {N}인 집합도 처리해야 합니다"


def test_sum_pooling_is_invariant():
    rng = np.random.default_rng(3)
    H = rng.normal(size=(1, 6, 4))
    Ws, Wp, b = params(rng)
    p = rng.permutation(6)
    s1 = equivariant_layer(H, Ws, Wp, b).sum(axis=1)
    s2 = equivariant_layer(H[:, p], Ws, Wp, b).sum(axis=1)
    assert np.allclose(s1, s2), "등변 층 출력을 원소 축으로 합하면 순서에 불변이어야 합니다"
