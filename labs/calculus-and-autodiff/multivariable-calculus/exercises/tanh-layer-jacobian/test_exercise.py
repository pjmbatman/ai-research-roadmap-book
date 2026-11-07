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


mlp_jacobian = load().mlp_jacobian


def g(W1, b1, W2, b2, x):
    return W2 @ np.tanh(W1 @ x + b1) + b2


def fd_jacobian(fun, x, eps=1e-6):
    return np.stack([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(x.size)], axis=1)


def params(n, k, m, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(k, n)), rng.normal(size=k), rng.normal(size=(m, k)), rng.normal(size=m), rng.normal(size=n)


@pytest.mark.parametrize("n,k,m", [(3, 4, 2), (1, 5, 1), (4, 2, 3)])
def test_matches_finite_difference(n, k, m):
    W1, b1, W2, b2, x = params(n, k, m, seed=n * 100 + k * 10 + m)
    J = mlp_jacobian(W1, b1, W2, b2, x)
    J_fd = fd_jacobian(lambda z: g(W1, b1, W2, b2, z), x)
    assert np.shape(J) == (m, n), f"Jacobian은 (출력 차원, 입력 차원) = ({m}, {n})이어야 합니다. 현재 {np.shape(J)}"
    assert np.allclose(J, J_fd, atol=1e-6), "유한차분 Jacobian과 다릅니다. 곱하는 순서와 tanh의 도함수를 확인하세요"


def test_output_bias_does_not_matter():
    W1, b1, W2, b2, x = params(3, 4, 2, seed=7)
    assert np.allclose(mlp_jacobian(W1, b1, W2, b2, x), mlp_jacobian(W1, b1, W2, b2 + 5.0, x)), \
        "출력 bias b2는 상수항이므로 Jacobian에 영향을 주지 않아야 합니다"


def test_saturated_units_block_gradient():
    W1, b1, W2, b2, x = params(3, 4, 2, seed=3)
    J = mlp_jacobian(W1, b1 + 50.0, W2, b2, x)        # 모든 은닉 뉴런이 포화
    assert np.allclose(J, 0.0, atol=1e-8), "tanh가 포화되면(입력이 매우 크면) Jacobian이 0에 가까워야 합니다"
