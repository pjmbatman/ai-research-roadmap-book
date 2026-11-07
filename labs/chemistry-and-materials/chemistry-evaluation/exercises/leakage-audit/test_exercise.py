import importlib.util
import os
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


leaked_fraction = load().leaked_fraction


def make(seed=0, n=400, L=48):
    rng = np.random.default_rng(seed)
    cores = (rng.random((30, L)) < 0.15)
    X = np.array([cores[rng.integers(30)].copy() for _ in range(n)])
    for x in X:
        for t in rng.choice(L, rng.integers(0, 2), replace=False):
            x[t] = True
    return X.astype(int), rng


def test_exact_copy_is_leak():
    X, rng = make()
    X[:50] = X[200:250]                              # test 앞 50개를 train 50개의 정확한 복사본으로
    tr = np.arange(200, 400)
    te = np.arange(0, 200)
    frac = leaked_fraction(X, tr, te, 1.0)
    assert frac >= 50 / 200 - 1e-9, "정확한 복사본(유사도 1)은 누수로 세어야 합니다"


def test_monotonic_in_threshold():
    X, _ = make(1)
    tr, te = np.arange(200, 400), np.arange(0, 200)
    fracs = [leaked_fraction(X, tr, te, t) for t in [0.3, 0.5, 0.7, 0.9, 1.0]]
    assert all(fracs[i] >= fracs[i + 1] - 1e-9 for i in range(len(fracs) - 1)), "임계가 높을수록 누수 비율은 줄어야 합니다"
    assert all(0.0 <= f <= 1.0 for f in fracs)


def test_disjoint_bits_no_leak():
    L = 48
    X = np.zeros((100, L), dtype=int)
    X[:50, :24] = 1                                  # train: 앞 절반 비트만
    X[50:, 24:] = 1                                  # test: 뒤 절반 비트만 (교집합 0)
    frac = leaked_fraction(X, np.arange(50), np.arange(50, 100), 0.1)
    assert frac == 0.0, "train과 겹치는 비트가 없으면 누수는 0이어야 합니다"
