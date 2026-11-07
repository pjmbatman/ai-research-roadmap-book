"""18.2 퀴즈(content/18-architectures/02-cnn/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/02-cnn/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}
rng = np.random.default_rng(0)


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def conv1d(x, w, s=1, p=0, d=1):                       # x (T,), w (k,) 직접 계산
    x = np.pad(x, p)
    span = d * (len(w) - 1) + 1
    return np.array([w @ x[t:t + span:d] for t in range(0, len(x) - span + 1, s)])


def test_conv_output_size():
    y = conv1d(rng.normal(size=50), rng.normal(size=5), s=2, p=1)
    assert len(y) == Q["conv-output-size"]["answer"] == 24


def test_receptive_field_with_stride():
    # 입력의 각 칸을 하나씩 바꿔 보며 출력 한 칸(가운데)이 영향을 받는 입력 범위를 직접 센다
    def net(x):
        h = x
        for s in (2, 2, 1):
            h = conv1d(h, np.ones(3), s=s)
        return h
    x0 = np.zeros(64)
    base = net(x0)
    j = len(base) // 2
    touched = [i for i in range(64) if net(x0 + np.eye(64)[i])[j] != base[j]]
    assert touched[-1] - touched[0] + 1 == Q["receptive-field-with-stride"]["answer"] == 15


def test_stride_breaks_equivariance():
    x, w = rng.normal(size=16), rng.normal(size=3)
    conv = lambda x, s: np.array([w @ np.roll(x, -t)[:3] for t in range(16)])[::s]
    y = conv(x, 2)
    shifts = lambda a: [np.roll(y, m) for m in range(len(y))]
    assert not any(np.allclose(conv(np.roll(x, 1), 2), z) for z in shifts(y))
    assert any(np.allclose(conv(np.roll(x, 2), 2), z) for z in shifts(y))
    assert correct("stride-breaks-equivariance") == [0]


def test_same_padding_dilated():
    p = Q["same-padding-dilated"]["answer"]
    assert len(conv1d(rng.normal(size=40), rng.normal(size=7), p=p, d=2)) == 40
    assert len(conv1d(rng.normal(size=40), rng.normal(size=7), p=3, d=2)) == 34


def test_effective_receptive_field():
    L = 64
    h = np.array([1.0])
    for _ in range(L):
        h = np.convolve(h, np.ones(3) / 3)
    m = next(m for m in range(L + 1) if h[L - m:L + m + 1].sum() >= 0.95)
    assert len(h) == 129 and 2 * m + 1 == 27
    assert correct("effective-receptive-field") == [0]


def test_unet_skip_role():
    assert correct("unet-skip-role") == [0]
