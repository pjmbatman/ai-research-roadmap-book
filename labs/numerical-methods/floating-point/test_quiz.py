"""9.1 퀴즈(content/09-numerical-methods/01-floating-point/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/09-numerical-methods/01-floating-point/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_machine_epsilon_meaning():
    eps = np.finfo(np.float64).eps
    assert 1.0 + eps > 1.0 and np.nextafter(1.0, 2.0) - 1.0 == eps          # 1 다음 수와의 간격
    assert np.finfo(np.float64).smallest_normal < eps * 1e-290             # 최소 양수와는 다른 양
    assert np.spacing(1e16) == 2.0                                           # 간격은 크기에 비례
    assert correct("machine-epsilon-meaning") == [1]


def test_float32_integer_gap():
    assert int(np.float32(2**24 + 1)) == Q["float32-integer-gap"]["answer"] == 2**24
    assert np.spacing(np.float32(2**24)) == 2.0


def test_bf16_vs_fp16():
    def fmt(e, m):                       # (최소 정규수, 최댓값, eps)
        bias = 2 ** (e - 1) - 1
        return 2.0 ** (1 - bias), (2 - 2.0**-m) * 2.0**bias, 2.0**-m
    fp16, bf16, fp32 = fmt(5, 10), fmt(8, 7), fmt(8, 23)
    assert fp16[:2] == (np.finfo(np.float16).smallest_normal, np.finfo(np.float16).max)
    assert bf16[0] == fp32[0]                                 # 같은 최소 정규수(같은 지수 비트)
    assert bf16[2] > fp16[2]                                  # bf16이 덜 정밀하다
    assert np.float16(1e-6) != 0 and np.float16(1e-6) < fp16[0]   # fp16에서는 subnormal(정밀도 손실)
    assert np.float16(1e-8) == 0                              # fp16 underflow
    assert correct("bf16-vs-fp16") == [0]


def test_cancellation_stable_form():
    x = 1e12
    exact = 1 / (np.sqrt(x + 1) + np.sqrt(x))
    naive = np.sqrt(x + 1) - np.sqrt(x)
    assert abs(naive - exact) / exact > 1e-6
    assert abs(exact - 5e-7) / 5e-7 < 1e-6
    x32 = np.float32(1e6)
    assert abs(float(np.sqrt(x32 + 1) - np.sqrt(x32)) - 5e-4) / 5e-4 > 0.02
    assert correct("cancellation-stable-form") == [1]


def test_softmax_shift():
    z = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    assert np.all(np.isnan(naive))
    stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    small = np.array([0.0, 1.0, 2.0])
    assert np.allclose(stable, np.exp(small) / np.exp(small).sum())
    assert np.allclose(np.round(stable, 3), [0.090, 0.245, 0.665])
    clipped = np.clip(z, -80, 80)
    assert np.allclose(np.exp(clipped) / np.exp(clipped).sum(), 1 / 3)       # clip은 답을 바꾼다
    assert np.nan_to_num(naive).sum() == 0                                    # nan_to_num은 합이 1이 아니다
    assert correct("softmax-shift") == [0]


def test_summation_order():
    small = np.full(10_000, 1e-4, dtype=np.float32)
    big_first = np.float32(1e4)
    for v in small:
        big_first += v
    small_first = np.float32(0)
    for v in small:
        small_first += v
    small_first += np.float32(1e4)
    assert float(big_first) == 1e4
    assert abs(float(small_first) - 10001.0) < 1e-2
    s, c = np.float32(1e4), np.float32(0)
    for v in small:                                                           # Kahan
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    assert abs(float(s) - 10001.0) < 2e-3
    assert correct("summation-order") == [0, 1]
