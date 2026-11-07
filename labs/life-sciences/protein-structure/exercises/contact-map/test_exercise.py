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


mod = load()
contact_map, top_l_precision = mod.contact_map, mod.top_l_precision


def hairpin():
    """두 개의 이상적인 α-나선을 나란히 놓은 Cα 좌표 (43, 3)."""
    def helix(n, start, sign):
        t = np.arange(n)
        a = np.deg2rad(100.0) * t
        return np.stack([2.3 * np.cos(a), 2.3 * np.sin(a), 1.5 * t * sign], 1) + start
    h1, h2 = helix(20, [0.0, 0.0, 0.0], 1), helix(20, [10.0, 0.0, 28.5], -1)
    loop = h1[-1] + np.linspace(0, 1, 5)[1:-1, None] * (h2[0] - h1[-1])
    return np.vstack([h1, loop, h2])


def test_simple_line():
    X = np.zeros((10, 3))
    X[:, 0] = np.arange(10) * 3.8                     # 일직선, 이웃 3.8 Å
    X[9] = X[0] + np.array([0.0, 5.0, 0.0])           # 마지막 잔기를 첫 잔기 옆으로
    C = contact_map(X)
    assert C.shape == (10, 10) and C.dtype == bool
    assert C[0, 9] and C[9, 0], "거리 5 Å, 간격 9인 쌍은 접촉입니다"
    assert not C[0, 1], "서열 간격이 min_sep보다 작은 쌍은 제외합니다"
    assert C.sum() == 4                                # (0, 9), (1, 9)와 대칭 쌍


def test_symmetric_no_diagonal():
    C = contact_map(hairpin())
    assert np.array_equal(C, C.T), "접촉 지도는 대칭입니다"
    assert not C.diagonal().any(), "대각선은 False입니다"


def test_cutoff_is_strict():
    X = np.zeros((8, 3))
    X[7] = [8.0, 0.0, 0.0]                             # 정확히 8 Å
    X[1:7, 1] = np.arange(1, 7) * 20.0                 # 나머지는 멀리
    assert not contact_map(X)[0, 7], "거리 < cutoff 이므로 정확히 8 Å은 접촉이 아닙니다"


def test_hairpin_contacts_between_helices():
    C = contact_map(hairpin())
    assert C.sum() // 2 == 22
    assert C[:20, 23:].any() and not C[:14, :14].any()


def test_top_l_precision_perfect_and_random():
    X = hairpin()
    C = contact_map(X)
    L = len(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    assert top_l_precision(-D, C) == 22 / L, "거리가 가까운 순서가 완벽한 예측이면 상위 L개 중 실제 접촉은 22개입니다"
    rng = np.random.default_rng(0)
    S = rng.random((L, L))
    S = (S + S.T) / 2
    p = top_l_precision(S, C)
    assert 0 <= p < 0.3, "무작위 점수의 정밀도는 낮아야 합니다"


def test_top_l_uses_only_long_range_pairs():
    X = hairpin()
    C = contact_map(X)
    L = len(X)
    S = np.zeros((L, L))
    idx = np.arange(L)
    S[np.abs(idx[:, None] - idx[None]) < 6] = 100.0    # 서열상 가까운 쌍에 큰 점수(무시되어야 함)
    S[C] = 1.0
    assert top_l_precision(S, C) == 22 / L, "|i-j| < min_sep인 쌍은 후보에서 빼야 합니다"
