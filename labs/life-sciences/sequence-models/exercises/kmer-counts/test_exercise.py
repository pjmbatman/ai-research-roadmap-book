import importlib.util
import os
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kmer_counts = load().kmer_counts


def reference(seq, k, alphabet):
    vocab = {"".join(p): i for i, p in enumerate(product(alphabet, repeat=k))}
    v = np.zeros(len(vocab))
    for s in range(len(seq) - k + 1):
        w = seq[s:s + k]
        if w in vocab:
            v[vocab[w]] += 1
    return v


def test_small_example():
    v = kmer_counts("ACGT", 2)
    assert np.shape(v) == (16,), f"길이는 4**2 = 16이어야 합니다. 현재 {np.shape(v)}"
    expected = np.zeros(16)
    expected[[1, 6, 11]] = 1                                 # AC, CG, GT
    assert np.array_equal(v, expected), "성분 순서는 AA, AC, AG, AT, CA, ... 사전식입니다"


def test_total_count():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 50))
    for k in [1, 2, 3]:
        assert kmer_counts(seq, k).sum() == 50 - k + 1


def test_protein_alphabet_matches_reference():
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list(aa), 40))
    assert np.array_equal(kmer_counts(seq, 2, aa), reference(seq, 2, aa))


def test_ambiguous_base_skipped():
    v = kmer_counts("ACNGT", 2)
    assert v.sum() == 2, "N이 들어간 k-mer(CN, NG)는 세지 않습니다"
    assert v[1] == 1 and v[11] == 1                         # AC, GT


def test_lowercase_and_short():
    assert np.array_equal(kmer_counts("acgt", 2), kmer_counts("ACGT", 2)), "소문자도 같은 결과여야 합니다"
    assert np.array_equal(kmer_counts("AC", 3), np.zeros(64)), "k보다 짧은 서열은 0 벡터입니다"
