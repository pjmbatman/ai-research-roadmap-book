"""22.2 퀴즈(content/22-experiment-operations/02-reproducibility/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/22-experiment-operations/02-reproducibility/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def test_float_order():
    big, one = np.float32(16777216), np.float32(1)
    assert (big + one) - big == np.float32(Q["float-order"]["answer"])
    assert (big - big) + one == np.float32(1)


def test_global_rng_shift():
    def order(extra):
        np.random.seed(0)
        np.random.normal(size=3)
        if extra:
            np.random.uniform(size=5)
        return np.random.permutation(6)

    def order_split(extra):
        init, data, aug = np.random.SeedSequence(0).spawn(3)
        np.random.default_rng(init).normal(size=3)
        if extra:
            np.random.default_rng(aug).uniform(size=5)
        return np.random.default_rng(data).permutation(6)

    assert not np.array_equal(order(False), order(True))           # 전역: 한 줄 추가로 순서가 바뀐다
    assert np.array_equal(order_split(False), order_split(True))   # 용도별: 바뀌지 않는다
    assert correct("global-rng-shift") == [0]


def test_choice_keys():
    assert correct("reproducibility-levels") == [2]
    assert correct("seed-not-enough") == [0, 1, 2]
    assert correct("deterministic-tradeoff") == [0]
    assert correct("when-determinism") == [0]
    assert correct("bitwise-across-machines") == [0]
