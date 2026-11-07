"""15.3 퀴즈(content/15-data/03-preprocessing-and-eda/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/15-data/03-preprocessing-and-eda/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_zscore_masking():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(10, 1, 90), rng.uniform(40, 60, 10)])
    z = (x - x.mean()) / x.std()
    mad = np.median(np.abs(x - np.median(x)))
    rz = 0.6745 * (x - np.median(x)) / mad
    assert x.std() > 8                                  # 이상치가 표준편차를 부풀린다
    assert np.sum(np.abs(z[90:]) > 3) < 10
    assert np.sum(np.abs(rz[90:]) > 3.5) == 10
    assert correct("zscore-masking") == [0]


def test_robust_z_value():
    q = Q["robust-z-value"]
    assert abs(0.6745 * (18.0 - 10.0) / 2.0 - q["answer"]) <= q["tolerance"]


def test_fit_on_train():
    assert correct("fit-on-train") == [0]


def test_smoothing_peak():
    t = np.arange(0, 60, 0.5)
    p = np.exp(-((t - 30.0) / 0.8) ** 2)
    sm = np.convolve(p, np.ones(9) / 9, mode="same")
    assert sm.max() < 0.5 * p.max()                       # 높이가 크게 준다
    assert abs(sm.sum() - p.sum()) < 1e-9                 # 넓이는 보존된다
    assert t[sm.argmax()] == t[p.argmax()]                # 위치는 그대로다
    assert correct("smoothing-peak") == [0]


def test_nyquist_resample():
    f_hz, fs_hz = 0.5, 0.8                                # 2f = 1 Hz보다 느린 표본화
    n = np.arange(400)
    x = np.sin(2 * np.pi * f_hz * n / fs_hz)
    spec = np.abs(np.fft.rfft(x))
    alias_hz = np.fft.rfftfreq(len(x), d=1 / fs_hz)[spec.argmax()]
    assert abs(alias_hz - 0.3) < 0.01                    # 0.5 Hz가 0.3 Hz로 보인다
    assert Q["nyquist-resample"]["answer"] == 2 * f_hz


def test_simpson_groups():
    rng = np.random.default_rng(0)
    xs, ys, within = [], [], []
    for x0, y0 in [(0, 6), (2, 4), (4, 2), (6, 0)]:
        x = x0 + rng.normal(0, 0.7, 50)
        y = y0 + 0.8 * (x - x0) + rng.normal(0, 0.3, 50)
        xs.append(x)
        ys.append(y)
        within.append(np.corrcoef(x, y)[0, 1])
    assert np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1] < -0.5
    assert min(within) > 0.5
    assert correct("simpson-groups") == [0]
