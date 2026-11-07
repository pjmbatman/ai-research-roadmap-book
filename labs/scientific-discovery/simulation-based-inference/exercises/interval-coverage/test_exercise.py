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


coverage = load().coverage


def conjugate_samples(M=2000, L=400, sd_factor=1.0, seed=0):
    """theta ~ N(0, 1), x ~ N(theta, 1). 정확한 사후분포 N(x/2, 1/2)의 표준편차에 sd_factor를 곱해 뽑는다."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=M)
    x = theta + rng.normal(size=M)
    sd = np.sqrt(0.5) * sd_factor
    samples = x[:, None] / 2 + sd * rng.normal(size=(M, L))   # (M, L)
    return samples, theta


@pytest.mark.parametrize("level", [0.5, 0.9])
def test_calibrated(level):
    samples, theta = conjugate_samples()
    c = coverage(samples, theta, level)
    assert abs(c - level) < 0.03, f"정확한 사후분포라면 coverage가 {level} 근처여야 합니다. 현재 {c:.3f}"


def test_overconfident():
    samples, theta = conjugate_samples(sd_factor=0.4)
    c = coverage(samples, theta, 0.9)
    assert c < 0.6, f"사후분포가 너무 좁으면 coverage가 크게 떨어져야 합니다. 현재 {c:.3f}"


def test_conservative():
    samples, theta = conjugate_samples(sd_factor=2.0)
    c = coverage(samples, theta, 0.9)
    assert c > 0.97, f"사후분포가 너무 넓으면 coverage가 명목값보다 높아야 합니다. 현재 {c:.3f}"


def test_exact_small_case():
    samples = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]])
    theta = np.array([2.0, 20.0])                           # 첫째는 구간 안, 둘째는 밖
    assert np.isclose(coverage(samples, theta, 0.5), 0.5), "중앙 50% 구간은 [1, 3]과 [11, 13]입니다"
