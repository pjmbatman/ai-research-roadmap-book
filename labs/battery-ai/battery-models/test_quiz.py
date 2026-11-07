"""30.2 퀴즈(content/30-battery-ai/02-battery-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

QUIZ = Path(__file__).parents[3] / "content/30-battery-ai/02-battery-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def ocv_v(soc):
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))


def sim(i_a, dt, r0, r1, c1, euler=False, soc0=0.8, q=3.0):
    a, v1, s, out = np.exp(-dt / (r1 * c1)), 0.0, soc0, []
    for i in i_a:
        out.append(ocv_v(s) - i * r0 - v1)
        v1 = v1 + dt * (-v1 / (r1 * c1) + i / c1) if euler else a * v1 + r1 * (1 - a) * i
        s -= i * dt / 3600 / q
    return np.array(out)


def test_ecm_instant_drop():
    v = sim(np.r_[0.0, np.full(10, 6.0)], 1.0, 0.025, 0.015, 2000.0)
    assert np.isclose((v[0] - v[1]) * 1e3, Q["ecm-instant-drop"]["answer"], atol=0.5)


def test_euler_step_instability():
    i = np.r_[np.zeros(1), np.full(9, 3.0)]
    v_eu = sim(i, 90.0, 0.02, 0.015, 2000.0, euler=True)
    v_ex = sim(i, 90.0, 0.02, 0.015, 2000.0)
    assert np.max(np.abs(v_eu - v_ex)) > 1.0          # Euler 발산
    assert abs(1 - 90 / 30) > 1 and 0 < np.exp(-3) < 1
    assert correct("euler-step-instability") == [0]


def test_cv_charging_dae_index1():
    # CV: OCV(SOC) - I R0 - V1 = Vmax 를 I에 대해 매 순간 풀 수 있다(R0 > 0)
    r0, r1, c1, vmax, dt = 0.02, 0.015, 2000.0, 4.1, 1.0
    a, v1, s, cur = np.exp(-dt / (r1 * c1)), 0.0, 0.85, []
    for _ in range(600):
        i = brentq(lambda i: ocv_v(s) - i * r0 - v1 - vmax, -100, 100)
        cur.append(i)
        v1 = a * v1 + r1 * (1 - a) * i
        s -= i * dt / 3600 / 3.0
    cur = np.array(cur)
    assert np.all(cur < 0) and np.all(np.diff(np.abs(cur)) < 0)   # 충전 전류(방전 + 규약에서 음수)가 줄어든다
    assert correct("cv-charging-dae") == [0]


def test_model_choice():
    assert correct("model-choice") == [0, 1]


def test_spm_assumption():
    assert correct("spm-assumption") == [0]


def test_ocv_slope_soc_error():
    assert np.isclose(5 / 2, Q["ocv-slope-soc-error"]["answer"])
