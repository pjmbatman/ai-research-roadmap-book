"""30.5 퀴즈(content/30-battery-ai/05-parameter-estimation/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/30-battery-ai/05-parameter-estimation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
THETA = np.array([0.020, 0.015, 2000.0])           # R0 [Ω], R1 [Ω], C1 [F]


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def eta(theta, i_a, dt_s=1.0):
    """1-RC 과전압 I R0 + V1 (영차 유지)."""
    r0, r1, c1 = theta
    a = np.exp(-dt_s / (r1 * c1))
    v1 = np.zeros_like(i_a)
    for k in range(len(i_a) - 1):
        v1[k + 1] = a * v1[k] + r1 * (1 - a) * i_a[k]
    return i_a * r0 + v1


def rel_sd(i_a, idx=None, sigma=1e-3):
    """참값에서 로그 파라미터의 상대 표준오차와 상관 행렬."""
    idx = np.arange(len(i_a)) if idx is None else idx
    f = lambda lt: eta(np.exp(lt), i_a)[idx]
    lt0, h = np.log(THETA), 1e-6
    S = np.column_stack([(f(lt0 + h * e) - f(lt0 - h * e)) / (2 * h) for e in np.eye(3)])
    cov = sigma**2 * np.linalg.inv(S.T @ S)
    sd = np.sqrt(np.diag(cov))
    return sd, cov / np.outer(sd, sd)


def test_r0_from_voltage_jump():
    i_a = np.where(np.arange(200) >= 10, 3.0, 0.0)
    e = eta(THETA, i_a)
    assert np.isclose(e[10] - e[9], 0.060)                     # 즉시 60 mV
    assert np.isclose(e[-1] - e[10], 0.045, atol=1e-4)         # 이후 45 mV
    r0_mohm = (e[10] - e[9]) / 3.0 * 1e3
    assert abs(r0_mohm - Q["r0-from-voltage-jump"]["answer"]) <= Q["r0-from-voltage-jump"]["tolerance"]


def test_short_pulse_design():
    t = np.arange(24)
    sd_short, _ = rel_sd(np.where(t >= 20, 3.0, 0.0))
    assert sd_short[1] > 10                                    # 2000% 수준
    t = np.arange(60)
    sd_rest, _ = rel_sd(np.where((t >= 20) & (t < 23), 3.0, 0.0))    # 같은 펄스 + 37 s 휴지
    assert 0.1 < sd_rest[1] < 0.2
    sd_long, _ = rel_sd(np.where(np.arange(400) >= 50, 3.0, 0.0))   # 펄스 350 s
    assert sd_long[1] < 0.05
    assert correct("short-pulse-design") == [0, 1]


def test_sampling_merges_r0_r1():
    t = np.arange(600)
    i_a = np.where((t >= 60) & (t < 360), 3.0, 0.0)
    _, corr_fast = rel_sd(i_a)
    sd_slow, corr_slow = rel_sd(i_a, np.arange(30, 600, 60))
    assert corr_slow[0, 1] < -0.99 < corr_fast[0, 1]
    assert np.all(sd_slow[:2] > 0.1)
    # 합 R0 + R1 방향의 상대오차는 작다: 로그 파라미터에서 합의 분산
    assert correct("sampling-merges-r0-r1") == [0]


def test_posterior_equals_prior():
    assert correct("posterior-equals-prior") == [0]


def test_hybrid_extrapolation():
    T = np.array([15.0, 20, 25, 30, 35])
    r0 = lambda Tc: 0.020 * np.exp(30e3 / 8.314 * (1 / (Tc + 273.15) - 1 / 298.15))
    meas = r0(T) * (1 + 0.02 * np.random.default_rng(1).normal(size=5))
    poly = np.polyval(np.polyfit(T, meas, 2), -10.0)
    arr = np.exp(np.polyval(np.polyfit(1 / (T + 273.15), np.log(meas), 1), 1 / 263.15))
    assert np.isclose(poly * 1e3, 76.6, atol=0.05) and np.isclose(arr * 1e3, 102.2, atol=0.05)
    assert abs(arr - r0(-10.0)) < abs(poly - r0(-10.0))
    assert correct("hybrid-extrapolation") == [0]


def test_surrogate_evaluation():
    assert correct("surrogate-evaluation") == [0, 1, 2]
