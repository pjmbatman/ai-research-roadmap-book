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


identity_clusters = load().identity_clusters


def ident_matrix(seqs):
    S = np.array([list(s) for s in seqs])
    return (S[:, None, :] == S[None, :, :]).mean(-1)


def synthetic(seed=0, n_sup=6, n_fam=3, n_mem=4, L=60):
    rng = np.random.default_rng(seed)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

    def evolve(s, rate):
        s = s.copy()
        hit = rng.random(L) < rate
        s[hit] = rng.choice(aa, hit.sum())
        return s

    out = []
    for _ in range(n_sup):
        root = rng.choice(aa, L)
        for _ in range(n_fam):
            anc = evolve(root, 0.35)
            out += ["".join(evolve(anc, 0.15)) for _ in range(n_mem)]
    return out


def test_small_example_numbering():
    seqs = ["AAAA", "CCCC", "AAAC", "CCCA", "GGGG"]
    labels = identity_clusters(seqs, 0.75)
    assert list(np.asarray(labels)) == [0, 1, 0, 1, 2], "번호는 처음 등장 순서대로 0, 1, 2, ..."


def test_threshold_inclusive():
    labels = identity_clusters(["AAAA", "AACC"], 0.5)
    assert labels[0] == labels[1], "동일성이 정확히 threshold인 쌍도 잇습니다(이상)"


def test_chaining():
    # 0–1, 1–2는 0.75 이상이지만 0–2는 0.5: 단일 연결이면 셋 다 한 군집
    seqs = ["AAAA", "AAAC", "AACC", "GGGG"]
    labels = list(np.asarray(identity_clusters(seqs, 0.75)))
    assert labels == [0, 0, 0, 1], "다른 서열을 거쳐 이어진 서열도 같은 군집입니다"


def test_cross_cluster_below_threshold():
    seqs = synthetic()
    t = 0.3
    labels = np.asarray(identity_clusters(seqs, t))
    I = ident_matrix(seqs)
    diff = labels[:, None] != labels[None, :]
    assert I[diff].max() < t, "서로 다른 군집의 서열 쌍은 모두 기준 미만이어야 합니다"


def test_clusters_are_connected_and_minimal():
    from scipy.sparse.csgraph import connected_components
    seqs = synthetic(seed=1)
    for t in [0.2, 0.3, 0.5]:
        labels = np.asarray(identity_clusters(seqs, t))
        n_ref = connected_components(ident_matrix(seqs) >= t, directed=False)[0]
        assert labels.max() + 1 == n_ref, f"기준 {t}: 군집 수가 연결 성분 수 {n_ref}와 달라요"
