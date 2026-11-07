"""27.3 퀴즈(content/27-earth-and-climate/03-downscaling-and-emulation/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/27-earth-and-climate/03-downscaling-and-emulation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

NS = {"__name__": "cells"}                    # 본문 셀의 결과로 문항의 근거를 확인한다
with contextlib.redirect_stdout(io.StringIO()):
    exec((HERE / "cells.py").read_text(encoding="utf-8"), NS)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_downscaling_not_super_resolution():
    # 같은 저해상도 값에 대해 고해상도 국지 변동이 남는다: 회귀 잔차의 분산이 0이 아니다
    assert NS["sd"].min() > 0.3
    assert correct("downscaling-not-super-resolution") == [0]


def test_regression_underestimates_extremes():
    obs, pred, X, te = NS["obs"], NS["pred"], NS["X"], NS["te"]
    d_obs, d_pred = obs - X[te], pred - X[te]
    assert (d_pred.std(0) / d_obs.std(0)).mean() < 0.5
    assert (np.percentile(pred, 99, axis=0) - np.percentile(obs, 99, axis=0)).mean() < -0.2
    assert correct("regression-underestimates-extremes") == [0]


def test_qm_wet_day_fraction():
    g = np.random.default_rng(5)
    n = 50000
    obs = np.where(g.random(n) < 0.3, g.gamma(0.7, 9.0, n), 0.0)
    mod = np.where(g.random(n) < 0.5, g.gamma(1.0, 3.5, n), 0.0)
    qs = np.linspace(0, 1, 1001)
    out = np.interp(mod, np.quantile(mod, qs), np.quantile(obs, qs))
    q = Q["qm-wet-day-fraction"]
    assert abs(np.mean(out > 0.1) - q["answer"]) <= q["tolerance"]


def test_qm_limits():
    qm, mod_h, mod_f, obs_h, mq = NS["qm"], NS["mod_h"], NS["mod_f"], NS["obs_h"], NS["mq"]
    p = lambda x: np.percentile(x, 99.9)
    raw = p(mod_f) / p(mod_h) - 1
    corr = p(qm(mod_f)) / p(qm(mod_h)) - 1
    assert corr > raw + 0.2                                   # 선택지 1: 변화 신호가 바뀐다
    assert qm(np.array([mq[-1] + 100]))[0] == obs_h.max()      # 선택지 2: 범위 밖은 끝값 고정
    assert np.isclose(np.percentile(qm(mod_h), 99), np.percentile(obs_h, 99), rtol=0.01)  # 선택지 3은 오답
    assert correct("qm-limits") == [0, 1, 2]


def test_emulator_extrapolation():
    emus, f_val, t_val, true_resp = NS["emus"], NS["f_val"], NS["t_val"], NS["true_resp"]
    r = {k: np.sqrt(np.mean((e(f_val) - t_val) ** 2)) for k, e in emus.items()}
    assert max(r.values()) - min(r.values()) < 0.03            # 범위 안에서는 비슷하다
    far = {k: float(np.atleast_1d(e(np.array([8.5])))[0]) for k, e in emus.items()}
    assert max(far.values()) - min(far.values()) > 4           # 범위 밖에서는 크게 갈린다
    assert far["k-NN"] < 0.5 * true_resp(8.5)
    assert correct("emulator-extrapolation") == [0]
