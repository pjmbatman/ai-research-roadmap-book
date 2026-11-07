"""18.5 퀴즈(content/18-architectures/05-language-models/quiz.toml) 정답을 계산으로 검증한다."""
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

QUIZ = Path(__file__).parents[3] / "content/18-architectures/05-language-models/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def _merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1]); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)


def _bpe(corpus, n):
    words = Counter(tuple(w) + ("</w>",) for w in corpus)
    merges = []
    for _ in range(n):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        words = Counter({_merge(w, best): c for w, c in words.items()})
    return merges


def _tok(word, merges):
    w = tuple(word) + ("</w>",)
    for p in merges:
        w = _merge(w, p)
    return list(w)


def test_bpe_number_split():
    corpus = "reaction reactor reactant react reacted 1990 1995 2000 2005 2010 2015 2020 2020 2021".split()
    merges = _bpe(corpus, 12)
    assert _tok("2020", merges) == ["2020</w>"]
    assert _tok("2019", merges) == ["201", "9", "</w>"]
    assert _tok("1920", merges) == ["1", "9", "20</w>"]
    assert _tok("2019", merges) == _tok("2019", merges)          # 결정적
    assert correct("bpe-number-split") == [0]


def test_held_out_perplexity():
    p = np.array([0.5, 0.25, 0.5, 0.125])
    ppl = np.exp(-np.log(p).mean())
    assert abs(ppl - Q["held-out-perplexity"]["answer"]) <= Q["held-out-perplexity"]["tolerance"]
    assert not np.isclose(1 / p.mean(), ppl, atol=0.05)          # 산술평균 오답과 구별


def test_train_ppl_below_floor():
    rng = np.random.default_rng(0)
    V, PERIOD, P = 4, 8, 0.9

    def make(n):
        x = rng.integers(V, size=n)
        for t in range(PERIOD, n):
            if rng.random() < P:
                x[t] = x[t - PERIOD]
        return x

    train, test = make(20_000), make(5_000)
    k = 8
    ids = lambda x: sum(x[i:len(x) - k + i] * V ** (k - 1 - i) for i in range(k))
    counts = np.zeros((V**k, V))
    np.add.at(counts, (ids(train), train[k:]), 1)
    prob = (counts + 0.1) / (counts + 0.1).sum(1, keepdims=True)
    ppl = lambda x: np.exp(-np.log(prob[ids(x), x[k:]]).mean())
    q = P + (1 - P) / V
    floor = np.exp(-(q * np.log(q) + (V - 1) * (1 - q) / V * np.log((1 - q) / V)))
    assert ppl(train) < floor < ppl(test)
    assert correct("train-ppl-below-floor") == [0]


def test_tokenizer_ppl_comparison():
    ppl_char_A = 12 ** (1 / 2)                                   # 토큰 수 = 글자 수 / 2
    assert np.isclose(ppl_char_A, 3.46, atol=0.01) and ppl_char_A > 3
    assert correct("tokenizer-ppl-comparison") == [0]


def test_temperature_effects():
    rng = np.random.default_rng(0)
    z = rng.normal(size=10)

    def dist(z, T=1.0, k=None):
        z = z / T
        if k is not None:
            z = np.where(z >= np.sort(z)[-k], z, -np.inf)
        p = np.exp(z - z.max())
        return p / p.sum()

    for T in [0.1, 0.5, 2.0, 10.0]:
        assert np.array_equal(np.argsort(dist(z, T)), np.argsort(z))     # 순위 불변
    assert dist(z, 1e-3)[np.argmax(z)] > 0.999                          # τ → 0: greedy
    assert np.isclose(dist(z, 1.0, k=1)[np.argmax(z)], 1.0)             # k = 1: greedy
    assert correct("temperature-effects") == [0, 1, 2]


def test_conceptual_answers():
    assert correct("training-stages") == [0]
    assert correct("rag-verification") == [0]
