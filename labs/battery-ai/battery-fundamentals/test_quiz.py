"""30.1 퀴즈(content/30-battery-ai/01-battery-fundamentals/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/30-battery-ai/01-battery-fundamentals/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ocv_v(soc):                       # 본문 cells.py와 같은 가상의 예시 셀
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))


def test_c_rate_discharge_time():
    q_ah, i_a = 2.5, 5.0
    c_rate = i_a / q_ah
    assert c_rate == 2.0
    assert 60 / c_rate == Q["c-rate-discharge-time"]["answer"] == 30


def test_energy_depends_on_rate():
    def energy(c, r=0.035, q=3.0, cut=3.4):
        i = c * q
        t = np.arange(0, 3600 / c, 1.0)
        v = ocv_v(1 - i * t / 3600 / q) - i * r
        n = np.argmax(v < cut) if np.any(v < cut) else len(t)
        return np.sum(v[:n] * i) / 3600, i * n / 3600, np.mean(v[:n])
    e_lo, q_lo, v_lo = energy(0.2)
    e_hi, q_hi, v_hi = energy(2.0)
    assert e_hi < e_lo and q_hi < q_lo and v_hi < v_lo     # 전하량도 평균 전압도 줄어든다
    assert correct("energy-depends-on-rate") == [0]


def test_soh_definition_mismatch():
    assert round(2.58 / 3.0 * 100, 1) == 86.0
    assert round(2.58 / 3.08 * 100, 1) == 83.8
    assert correct("soh-definition-mismatch") == [0]


def test_efc_partial_cycles():
    efc = 300 * (0.70 - 0.20) * 3.0 / 3.0
    assert np.isclose(efc, Q["efc-partial-cycles"]["answer"])


def test_sign_convention_check():
    # 충전 +로 기록한 3 A 정전류 구간: SOC가 오르므로 전압이 오른다
    t = np.arange(600.0)
    soc = 0.3 + 3.0 * t / 3600 / 3.0
    v = ocv_v(soc) + 3.0 * 0.035
    assert np.all(np.diff(v) > 0)
    assert correct("sign-convention-check") == [0]


def test_knee_linear_extrapolation():
    n = np.arange(0, 1501)
    cap = 3.08 - 2.0e-4 * n - 0.02 * np.exp((n - 900) / 120)
    eol_true = n[np.argmax(cap < 2.4)]
    slope, icpt = np.polyfit(n[:400], cap[:400], 1)
    eol_lin = (2.4 - icpt) / slope
    assert 1250 < eol_true < 1290 and 3300 < eol_lin < 3500
    # 2차 다항식 외삽도 knee 시점을 맞히지 못한다
    c2 = np.polyfit(n[:400], cap[:400], 2)
    roots = np.roots(c2 - np.array([0, 0, 2.4]))
    real = roots[np.isreal(roots)].real
    assert not np.any(np.abs(real[real > 0] - eol_true) < 100)
    assert correct("knee-linear-extrapolation") == [0]
