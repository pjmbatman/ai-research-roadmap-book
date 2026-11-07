import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def random_graph(n, seed):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.3).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def test_sums_to_one():
    r = m.pagerank(random_graph(12, 0))
    assert r.shape == (12,), "반환값의 shape은 (n,)이어야 합니다"
    assert r.sum() == pytest.approx(1.0, abs=1e-9), "PageRank의 합은 1이어야 합니다"
    assert np.all(r > 0), "damping 때문에 모든 논문은 양의 값을 받습니다"


def test_symmetric_cycle_is_uniform():
    n = 5
    A = np.zeros((n, n))
    for i in range(n):
        A[(i + 1) % n, i] = 1.0                           # i가 i+1을 인용하는 순환
    assert m.pagerank(A) == pytest.approx(np.full(n, 1 / n), abs=1e-9)


def test_fixed_point():
    A, d = random_graph(10, 1), 0.85
    r = m.pagerank(A, d=d)
    n, out = len(r), A.sum(axis=0)
    P = np.where(out > 0, A / np.where(out > 0, out, 1.0), 1.0 / n)
    assert r == pytest.approx((1 - d) / n + d * P @ r, abs=1e-9), "r = (1-d)/n + d P r를 만족해야 합니다"


def test_dangling_papers():
    A = np.zeros((4, 4))
    A[0, 1] = A[0, 2] = 1.0                               # 1, 2가 0을 인용. 0과 3은 참고문헌이 없다
    r = m.pagerank(A)
    assert np.all(np.isfinite(r)), "참고문헌이 없는 논문의 열을 균등분포로 채우세요"
    assert r.sum() == pytest.approx(1.0, abs=1e-9)
    assert r[0] > r[1] == pytest.approx(r[2]) and r[1] == pytest.approx(r[3])


def test_cited_by_important_paper_wins():
    # 0과 1은 둘 다 한 번씩 인용된다. 0을 인용한 4는 많이 인용되는 논문이고, 1을 인용한 5는 아무도 인용하지 않는다.
    A = np.zeros((6, 6))
    A[0, 4] = A[1, 5] = 1.0
    A[4, 2] = A[4, 3] = 1.0
    r = m.pagerank(A)
    assert A[0].sum() == A[1].sum()
    assert r[0] > r[1], "피인용 수가 같아도 중요한 논문에게 인용되면 PageRank가 더 높습니다"


def test_damping_zero_is_uniform():
    assert m.pagerank(random_graph(7, 2), d=0.0) == pytest.approx(np.full(7, 1 / 7))
