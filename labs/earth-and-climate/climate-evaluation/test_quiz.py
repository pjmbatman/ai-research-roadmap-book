"""27.4 퀴즈(content/27-earth-and-climate/04-climate-evaluation/quiz.toml) 정답을 계산으로 검증한다."""
import contextlib
import io
import tomllib
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/27-earth-and-climate/04-climate-evaluation/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

NS = {"__name__": "cells"}                    # 본문 셀의 결과로 문항의 근거를 확인한다
with contextlib.redirect_stdout(io.StringIO()):
    exec((HERE / "cells.py").read_text(encoding="utf-8"), NS)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_acc_ignores_amplitude():
    acc, sharp, blurry, truth = NS["acc"], NS["sharp"], NS["blurry"], NS["truth"]
    assert np.isclose(acc(0.5 * sharp), acc(sharp))
    r = lambda f: np.sqrt(((f - truth) ** 2).mean())
    assert r(0.5 * sharp) < r(sharp)                       # 예측력이 약한 예보는 줄이면 좋아지고
    assert r(0.5 * truth) > r(truth)                       # 정확한 예보는 줄이면 나빠진다
    assert correct("acc-ignores-amplitude") == [0]


def test_underdispersed_ensemble():
    ens = NS["ensemble"](0.5)
    truth, M = NS["truth"], NS["M"]
    rm = np.sqrt(((ens.mean(0) - truth) ** 2).mean())
    sp = np.sqrt(ens.var(0, ddof=1).mean() * (M + 1) / M)
    hist = np.bincount((ens < truth).sum(0).ravel(), minlength=M + 1) / truth.size
    assert sp / rm < 0.6 and hist[0] + hist[-1] > 2 * 2 / (M + 1)
    assert abs(NS["crps"](ens) - NS["crps"](NS["ensemble"](1.0))) < 0.05   # CRPS는 둔감
    assert correct("underdispersed-ensemble") == [0]


def test_blurry_wins_rmse():
    sharp, blurry, truth, power = NS["sharp"], NS["blurry"], NS["truth"], NS["power"]
    r = lambda f: np.sqrt(((f - truth) ** 2).mean())
    assert r(blurry) < r(sharp) and abs(r(sharp) / r(blurry) - np.sqrt(2)) < 0.1
    assert NS["acc"](blurry) > NS["acc"](sharp)
    assert power(blurry)[16] / power(truth)[16] < 0.05
    assert correct("blurry-wins-rmse") == [0]


def test_effective_sample_size():
    q = Q["effective-sample-size"]
    assert abs(100 * 0.4 / 1.6 - q["answer"]) <= q["tolerance"]
    assert NS["np"].mean(NS["np"].abs(NS["t_naive"]) > NS["crit"]) > 0.25


def test_return_period_in_30_years():
    q = Q["return-period-in-30-years"]
    p = 1 - 0.99**30
    assert abs(p - q["answer"]) <= q["tolerance"]
    assert abs(0.30 - q["answer"]) > q["tolerance"]
    level, ref = NS["level"], NS["ref"]
    assert 25 <= 1 / np.mean(ref + 1.0 > level) <= 40       # 본문: +1 °C에서 재현 기간 31년


def test_gev_short_record():
    g = np.random.default_rng(11)
    rls = []
    for _ in range(3):
        c, loc, scale = stats.genextreme.fit(NS["annual_max"](g, 30))
        rls.append(stats.genextreme.ppf(0.99, c, loc, scale))
    assert max(rls) - min(rls) > 0.5                       # 같은 기후의 30년 표본끼리도 갈린다
    # scipy의 c > 0은 상한이 있는 분포(ξ < 0)
    assert np.isfinite(stats.genextreme(0.2).support()[1]) and np.isinf(stats.genextreme(-0.2).support()[1])
    assert correct("gev-short-record") == [0, 1]
