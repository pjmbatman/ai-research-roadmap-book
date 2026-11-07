import importlib.util
import math
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


m = load()
LOG_Z, MU, SD = -3.0, 2.0, 0.5                         # 사후 N(2, 0.5^2), log p(D) = -3


def log_joint(theta):
    return LOG_Z - 0.5 * ((theta - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi))


def test_kl_known_values():
    assert m.kl_gaussians(0, 1, 0, 1) == pytest.approx(0.0)
    assert m.kl_gaussians(0, 1, 0, 2) == pytest.approx(math.log(2) + 1 / 8 - 0.5)
    assert m.kl_gaussians(0, 2, 0, 1) == pytest.approx(-math.log(2) + 2 - 0.5), "KL은 방향에 따라 값이 다릅니다"
    assert m.kl_gaussians(1, 1, 0, 1) == pytest.approx(0.5)


def test_elbo_exact_when_q_is_posterior():
    eps = np.random.default_rng(0).normal(size=5)
    assert m.elbo_mc(log_joint, MU, SD, eps) == pytest.approx(LOG_Z), "q가 사후분포와 같으면 ELBO = log p(D)이고 추정 분산이 0입니다"


def test_elbo_plus_kl_is_log_evidence():
    eps = np.random.default_rng(1).normal(size=200000)
    for mq, sq in [(1.0, 0.3), (2.5, 1.0), (2.0, 0.1)]:
        e = m.elbo_mc(log_joint, mq, sq, eps)
        assert e + m.kl_gaussians(mq, sq, MU, SD) == pytest.approx(LOG_Z, abs=0.02)
        assert e < LOG_Z, "ELBO는 log p(D)보다 클 수 없습니다"


def test_accepts_list_eps():
    assert m.elbo_mc(log_joint, MU, SD, [0.0, 1.0, -1.0]) == pytest.approx(LOG_Z)


def test_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        m.elbo_mc(log_joint, 0.0, 0.0, [0.0])
