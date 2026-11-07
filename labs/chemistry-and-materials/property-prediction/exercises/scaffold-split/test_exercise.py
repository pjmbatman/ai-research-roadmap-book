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


scaffold_split = load().scaffold_split


def make_scaffolds(seed, n=500, n_scaf=60):
    rng = np.random.default_rng(seed)
    p = 1 / np.arange(1, n_scaf + 1)
    return [f"scaf{k}" for k in rng.choice(n_scaf, n, p=p / p.sum())]


def test_partition():
    scafs = make_scaffolds(0)
    tr, te = scaffold_split(scafs)
    assert sorted(np.concatenate([tr, te]).tolist()) == list(range(len(scafs))), "모든 분자가 정확히 한 번씩 train이나 test에 들어가야 합니다"


def test_no_shared_scaffold():
    for seed in range(3):
        scafs = np.array(make_scaffolds(seed))
        tr, te = scaffold_split(scafs)
        shared = set(scafs[tr]) & set(scafs[te])
        assert not shared, f"train과 test가 골격을 공유합니다: {sorted(shared)[:3]}"


def test_largest_scaffold_in_train():
    scafs = np.array(make_scaffolds(1))
    tr, _ = scaffold_split(scafs)
    labels, counts = np.unique(scafs, return_counts=True)
    assert labels[counts.argmax()] in set(scafs[tr]), "가장 큰 골격은 train에 들어가야 합니다"


def test_fraction_close():
    scafs = make_scaffolds(2, n=1000, n_scaf=200)
    tr, te = scaffold_split(scafs, test_frac=0.2)
    assert 0.18 <= len(te) / len(scafs) <= 0.25, f"test 비율이 목표 0.2에서 너무 멉니다: {len(te) / len(scafs):.3f}"


def test_small_example_exact():
    scafs = ["a", "b", "a", "c", "b", "a", "d", "c", "a", "e"]   # 크기: a 4, b 2, c 2, d 1, e 1
    tr, te = scaffold_split(scafs, test_frac=0.3)               # train 최대 7개: a(4) + b(2) = 6, c는 넘침, d(1) = 7
    assert tr.tolist() == [0, 1, 2, 4, 5, 6, 8], f"train 인덱스가 다릅니다: {tr.tolist()}"
    assert te.tolist() == [3, 7, 9], f"test 인덱스가 다릅니다: {te.tolist()}"
