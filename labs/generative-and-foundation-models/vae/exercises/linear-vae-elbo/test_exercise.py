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


elbo_terms = load().elbo_terms


def log_px(x, w, b, sigma):
    C = np.outer(w, w) + sigma**2 * np.eye(len(x))
    d = x - b
    return -0.5 * (d @ np.linalg.solve(C, d) + np.log(np.linalg.det(2 * np.pi * C)))


def test_kl_zero_when_q_is_prior():
    _, kl = elbo_terms(np.zeros(2), np.ones(2), np.zeros(2), 1.0, 0.0, 1.0)
    assert np.isclose(kl, 0.0), f"q = N(0, 1)이면 KL은 0입니다. 현재 {kl}"


def test_matches_monte_carlo():
    rng = np.random.default_rng(0)
    x, w, b = np.array([1.0, -0.5, 0.3]), np.array([0.8, 0.2, -1.1]), np.array([0.1, 0.0, -0.2])
    sigma, mu_q, s_q = 0.7, 0.4, 0.5
    rec, kl = elbo_terms(x, w, b, sigma, mu_q, s_q)
    z = mu_q + s_q * rng.normal(size=200_000)
    r = x[None] - b - z[:, None] * w
    rec_mc = np.mean(-1.5 * np.log(2 * np.pi * sigma**2) - (r**2).sum(1) / (2 * sigma**2))
    log_q = -0.5 * ((z - mu_q) / s_q) ** 2 - np.log(s_q * np.sqrt(2 * np.pi))
    log_p = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    assert abs(rec - rec_mc) < 0.02, f"재구성 항: 닫힌 꼴 {rec:.4f}, Monte Carlo {rec_mc:.4f}"
    assert abs(kl - np.mean(log_q - log_p)) < 0.01, "KL 항이 Monte Carlo 추정과 다릅니다"


def test_exact_posterior_gives_log_evidence():
    x, w, b, sigma = np.array([1.4, -0.9]), np.array([1.5, -0.8]), np.array([0.2, 0.1]), 0.5
    s2 = sigma**2 / (sigma**2 + w @ w)
    m = w @ (x - b) / (sigma**2 + w @ w)
    rec, kl = elbo_terms(x, w, b, sigma, m, np.sqrt(s2))
    assert np.isclose(rec - kl, log_px(x, w, b, sigma)), "q가 정확한 사후분포이면 ELBO = log p(x)입니다"


def test_elbo_is_lower_bound():
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, w, b = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        sigma, mu_q, s_q = rng.uniform(0.3, 2), rng.normal(), rng.uniform(0.1, 2)
        rec, kl = elbo_terms(x, w, b, sigma, mu_q, s_q)
        assert rec - kl <= log_px(x, w, b, sigma) + 1e-9, "ELBO는 log p(x)를 넘을 수 없습니다"
