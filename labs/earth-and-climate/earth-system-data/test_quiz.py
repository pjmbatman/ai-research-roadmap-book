"""27.1 퀴즈(content/27-earth-and-climate/01-earth-system-data/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/27-earth-and-climate/01-earth-system-data/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
LAT = np.arange(-89.75, 90, 0.5)                      # (360,) 0.5도 격자 칸 중심
W = np.cos(np.deg2rad(LAT))


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_reanalysis_is_not_observation():
    assert correct("reanalysis-is-not-observation") == [1]


def test_polar_cap_area():
    frac = W[np.abs(LAT) > 60].sum() / W.sum()
    q = Q["polar-cap-area"]
    assert abs(frac - (1 - np.sin(np.pi / 3))) < 1e-4
    assert abs(frac - q["answer"]) <= q["tolerance"]
    assert abs(np.mean(np.abs(LAT) > 60) - q["answer"]) > q["tolerance"]   # 칸 개수 비율(1/3)은 오답


def test_unweighted_mean_is_too_cold():
    t = 300 - 45 * np.sin(np.deg2rad(LAT)) ** 2       # 경도 방향으로 같으므로 위도만 보면 된다
    assert t.mean() < (W * t).sum() / W.sum() - 5
    assert correct("unweighted-mean-bias") == [0]


def test_accumulated_precip_rate():
    acc_mm = np.array([0.0, 2.0, 5.5, 5.5, 9.0])      # 0, 6, 12, 18, 24시
    rate = (acc_mm[4] - acc_mm[2]) / 12
    q = Q["accumulated-precip-rate"]
    assert abs(rate - q["answer"]) <= q["tolerance"]
    assert abs(acc_mm[4] / 24 - q["answer"]) > q["tolerance"]


def test_missing_values_in_mean():
    lon = np.arange(0, 360, 1.0)
    sst = np.repeat((300 - 45 * np.sin(np.deg2rad(LAT)) ** 2)[:, None], lon.size, axis=1)
    sst[(LAT[:, None] > 0) & (LAT[:, None] < 70) & (lon[None, :] < 140)] = np.nan
    w = W[:, None] * np.ones(lon.size)
    ok = ~np.isnan(sst)
    good = (w * sst)[ok].sum() / w[ok].sum()
    zero = (w * np.nan_to_num(sst)).sum() / w.sum()
    assert zero < good - 30 and abs(np.nanmean(sst) - good) > 1
    assert correct("missing-values-in-mean") == [2]


def test_anomaly_base_period():
    rng = np.random.default_rng(0)
    years = np.arange(1981, 2021)
    month, year = np.tile(np.arange(12), years.size), np.repeat(years, 12)
    t = -10 * np.cos(2 * np.pi * (month + 0.5) / 12) + 0.03 * (year - 1981) + rng.normal(0, 0.8, year.size)
    base = year >= 1991
    clim = np.array([t[base & (month == m)].mean() for m in range(12)])
    anom = t - clim[month]
    assert abs(anom[base].mean()) < 1e-10                      # 선택지 0
    clim2 = np.array([t[(year < 2011) & (month == m)].mean() for m in range(12)])
    assert abs(np.mean(clim - clim2)) > 0.2                    # 선택지 1: 기준 기간이 다르면 편차가 이동
    assert anom[~base].mean() < -0.3                           # 선택지 2(오답): 추세가 남는다
    assert correct("anomaly-base-period") == [0, 1]
