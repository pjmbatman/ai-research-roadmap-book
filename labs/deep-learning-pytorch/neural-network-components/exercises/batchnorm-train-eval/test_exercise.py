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


batchnorm_forward = load().batchnorm_forward


def setup(B=6, D=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(B, D)) * np.array([1.0, 5.0, 0.5]) + np.array([2.0, -3.0, 10.0])
    gamma, beta = rng.normal(size=D), rng.normal(size=D)
    rm, rv = rng.normal(size=D), rng.uniform(0.5, 2.0, size=D)
    return x, gamma, beta, rm, rv


def test_train_normalizes_with_batch_stats():
    x, gamma, beta, rm, rv = setup()
    y, _, _ = batchnorm_forward(x, np.ones(3), np.zeros(3), rm, rv, training=True)
    assert np.allclose(y.mean(axis=0), 0, atol=1e-8), "train 모드 출력의 특성별 평균은 0이어야 합니다"
    assert np.allclose(y.var(axis=0), 1, atol=1e-3), "train 모드 출력의 특성별 (편향) 분산은 1이어야 합니다"
    y2, _, _ = batchnorm_forward(x, gamma, beta, rm, rv, training=True)
    assert np.allclose(y2, gamma * y + beta), "y = gamma * x_hat + beta여야 합니다"


def test_running_stats_update_uses_unbiased_variance():
    x, gamma, beta, rm, rv = setup()
    _, nm, nv = batchnorm_forward(x, gamma, beta, rm, rv, training=True, momentum=0.1)
    assert np.allclose(nm, 0.9 * rm + 0.1 * x.mean(axis=0)), "running_mean 갱신 규칙이 다릅니다"
    assert np.allclose(nv, 0.9 * rv + 0.1 * x.var(axis=0, ddof=1)), \
        "running_var는 불편 분산(B-1로 나눔)으로 갱신해야 합니다"


def test_eval_uses_running_stats_and_does_not_update():
    x, gamma, beta, rm, rv = setup()
    y, nm, nv = batchnorm_forward(x, gamma, beta, rm, rv, training=False)
    assert np.allclose(y, gamma * (x - rm) / np.sqrt(rv + 1e-5) + beta), "eval 모드는 running 통계로 정규화해야 합니다"
    assert np.allclose(nm, rm) and np.allclose(nv, rv), "eval 모드에서는 통계를 갱신하지 않습니다"


def test_eval_is_independent_of_batch_composition():
    x, gamma, beta, rm, rv = setup()
    y_all, _, _ = batchnorm_forward(x, gamma, beta, rm, rv, training=False)
    y_one, _, _ = batchnorm_forward(x[2:3], gamma, beta, rm, rv, training=False)
    assert np.allclose(y_all[2:3], y_one), "eval 모드의 출력은 같은 batch의 다른 표본과 무관해야 합니다"


def test_train_depends_on_batch_composition():
    x, gamma, beta, rm, rv = setup()
    y_all, _, _ = batchnorm_forward(x, gamma, beta, rm, rv, training=True)
    y_half, _, _ = batchnorm_forward(x[:3], gamma, beta, rm, rv, training=True)
    assert not np.allclose(y_all[:3], y_half), "train 모드 출력은 batch 통계에 따라 달라져야 합니다"


def test_single_sample_train_raises():
    x, gamma, beta, rm, rv = setup()
    with pytest.raises(ValueError):
        batchnorm_forward(x[:1], gamma, beta, rm, rv, training=True)


def test_inputs_not_modified():
    x, gamma, beta, rm, rv = setup()
    copies = [a.copy() for a in (x, rm, rv)]
    batchnorm_forward(x, gamma, beta, rm, rv, training=True)
    for a, c in zip((x, rm, rv), copies):
        assert np.array_equal(a, c), "입력 배열(x, running_mean, running_var)을 제자리에서 바꾸면 안 됩니다"
