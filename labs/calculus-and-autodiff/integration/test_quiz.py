"""8.3 퀴즈(content/08-calculus-and-autodiff/03-integration/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy import integrate

QUIZ = Path(__file__).parents[3] / "content/08-calculus-and-autodiff/03-integration/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_iterated_integral_limits():
    f = lambda x, y: np.exp(x) * np.cos(3 * y) + x * y          # 대칭이 아닌 함수
    ref, _ = integrate.dblquad(lambda y, x: f(x, y), 0, 1, lambda x: 0, lambda x: x)
    good, _ = integrate.dblquad(lambda x, y: f(x, y), 0, 1, lambda y: y, lambda y: 1)
    other, _ = integrate.dblquad(lambda x, y: f(x, y), 0, 1, lambda y: 0, lambda y: y)
    square, _ = integrate.dblquad(lambda x, y: f(x, y), 0, 1, lambda y: 0, lambda y: 1)
    assert np.isclose(ref, good) and not np.isclose(ref, other) and not np.isclose(ref, square)
    assert correct("iterated-integral-limits") == [0]


def test_triangle_integral():
    val, _ = integrate.dblquad(lambda y, x: x * y, 0, 1, lambda x: 0, lambda x: x)
    q = Q["triangle-integral"]
    assert abs(val - q["answer"]) <= q["tolerance"]


def test_det_area_scaling():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ A.T
    x, y = corners[:, 0], corners[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert np.isclose(shoelace, Q["det-area-scaling"]["answer"])


def test_polar_jacobian():
    area_r, _ = integrate.dblquad(lambda r, th: r, 0, 2 * np.pi, lambda th: 0, lambda th: 1)
    area_no_r, _ = integrate.dblquad(lambda r, th: 1.0, 0, 2 * np.pi, lambda th: 0, lambda th: 1)
    assert np.isclose(area_r, np.pi) and np.isclose(area_no_r, 2 * np.pi)
    assert correct("polar-jacobian") == [0]


def test_gaussian_normalizer():
    Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])
    P = np.linalg.inv(Sigma)
    val, _ = integrate.dblquad(lambda y, x: np.exp(-0.5 * np.array([x, y]) @ P @ np.array([x, y])),
                               -12, 12, lambda x: -12, lambda x: 12)
    d = 2
    assert np.isclose(val, (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(Sigma)), rtol=1e-6)
    assert not np.isclose(val, (2 * np.pi) ** (d / 2) / np.sqrt(np.linalg.det(Sigma)), rtol=1e-3)
    assert correct("gaussian-normalizer") == [0]


def test_density_transform_scale():
    y = 3 * rng.uniform(size=2_000_000) + 1
    h = 0.05
    est = np.mean(np.abs(y - 2) < h) / (2 * h)
    q = Q["density-transform-scale"]
    assert abs(est - q["answer"]) <= q["tolerance"]


def test_flow_logdet_sign():
    # 1차원 flow x = f(z) = exp(z), z ~ N(0, 1): 공식 log p_Z(z) - log|f'(z)|이 로그정규 밀도와 같다
    z = rng.normal(size=5)
    x = np.exp(z)
    log_pz = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    lognormal = -np.log(x) - 0.5 * np.log(x) ** 2 - 0.5 * np.log(2 * np.pi)
    assert np.allclose(log_pz - z, lognormal)            # log|f'(z)| = z
    assert not np.allclose(log_pz + z, lognormal)
    assert correct("flow-logdet-sign") == [0]
