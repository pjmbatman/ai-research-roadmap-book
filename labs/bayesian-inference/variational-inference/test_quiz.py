"""13.6 퀴즈(content/13-bayesian-inference/06-variational-inference/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

QUIZ = Path(__file__).parents[3] / "content/13-bayesian-inference/06-variational-inference/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
x = np.linspace(-10, 10, 8001)
dx = x[1] - x[0]


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def log_n(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))


def test_elbo_gap_is_reverse_kl():
    # 비정규화 목표: Z * (0.3 N(-1, 0.5) + 0.7 N(1.5, 1)), log Z = 2
    p = 0.3 * np.exp(log_n(x, -1, 0.5)) + 0.7 * np.exp(log_n(x, 1.5, 1.0))
    log_joint = 2.0 + np.log(p)
    q = np.exp(log_n(x, 1.0, 0.8))
    elbo = np.sum(q * (log_joint - np.log(q))) * dx
    kl_rev = np.sum(q * (np.log(q) - np.log(p))) * dx
    assert abs(elbo + kl_rev - 2.0) < 1e-6 and elbo < 2.0
    assert correct("elbo-gap") == [0]


def test_elbo_numeric():
    assert abs((-420.1 - 3.1) - Q["elbo-numeric"]["answer"]) < 1e-9


def test_reparameterization_gradient_matches():
    # d/dm E_q[f(theta)], f = -theta^4: 재매개변수화 추정 vs 해석해 -(4m^3 + 12 m s^2)
    rng = np.random.default_rng(0)
    m, s, eps = 0.7, 0.4, rng.normal(size=400000)
    g = np.mean(-4 * (m + s * eps) ** 3)
    assert abs(g - (-(4 * m**3 + 12 * m * s**2))) < 0.02
    assert correct("reparameterization-purpose") == [0]


def fit(direction, m0, p):
    def obj(v):
        q = np.exp(log_n(x, v[0], np.exp(v[1])))
        a, b = (q, p) if direction == "reverse" else (p, q)
        return np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx
    v = minimize(obj, [m0, 0.0], method="Nelder-Mead").x
    return v[0], np.exp(v[1])


def test_reverse_kl_is_mode_seeking():
    p = 0.5 * np.exp(log_n(x, -2, 0.6)) + 0.5 * np.exp(log_n(x, 2, 0.6))
    m_r, s_r = fit("reverse", 0.5, p)
    m_f, s_f = fit("forward", 0.5, p)
    assert abs(abs(m_r) - 2) < 0.05 and s_r < 0.7       # 한 봉우리
    assert abs(m_f) < 0.05 and s_f > 2.0                 # 두 봉우리를 덮음
    assert correct("reverse-kl-bimodal") == [0]


def test_mean_field_sd():
    rho = 0.8
    Lam = np.linalg.inv(np.array([[1, rho], [rho, 1]]))
    Sigma = np.linalg.inv(Lam)

    def kl_q_p(log_s):                                    # KL(N(0, diag s^2) || N(0, Sigma))
        S = np.diag(np.exp(2 * log_s))
        return 0.5 * (np.trace(Lam @ S) - 2 + np.log(np.linalg.det(Sigma)) - np.log(np.linalg.det(S)))

    sd = np.exp(minimize(kl_q_p, [0.0, 0.0]).x)
    assert np.allclose(sd, Q["mean-field-sd"]["answer"], atol=1e-4)


def test_laplace_sd():
    assert abs(1 / np.sqrt(400) - Q["laplace-sd"]["answer"]) < 1e-12


def test_laplace_when_poor():
    assert correct("laplace-when-poor") == [0, 1]
