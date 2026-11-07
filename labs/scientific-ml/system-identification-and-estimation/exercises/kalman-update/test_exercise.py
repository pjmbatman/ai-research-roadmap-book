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


kf_update = load().kf_update


def random_problem(n, k, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    P = A @ A.T + 0.5 * np.eye(n)
    B = rng.normal(size=(k, k))
    R = B @ B.T + 0.1 * np.eye(k)
    return rng.normal(size=n), P, rng.normal(size=k), rng.normal(size=(k, n)), R


def test_matches_gaussian_conditioning():
    for n, k, seed in [(2, 1, 0), (4, 2, 1), (3, 3, 2)]:
        m, P, y, H, R = random_problem(n, k, seed)
        # 결합분포 (x, y) ~ N([m, Hm], [[P, P H^T], [H P, H P H^T + R]])에서 x | y
        Sxy, Syy = P @ H.T, H @ P @ H.T + R
        m_ref = m + Sxy @ np.linalg.solve(Syy, y - H @ m)
        P_ref = P - Sxy @ np.linalg.solve(Syy, Sxy.T)
        m_new, P_new = kf_update(m, P, y, H, R)
        assert np.shape(m_new) == (n,) and np.shape(P_new) == (n, n), "반환 shape이 (n,), (n, n)이어야 합니다"
        assert np.allclose(m_new, m_ref), "갱신된 평균이 가우시안 조건부 평균과 다릅니다"
        assert np.allclose(P_new, P_ref), "갱신된 공분산이 가우시안 조건부 공분산과 다릅니다"


def test_scalar_case():
    m_new, P_new = kf_update(np.array([0.0]), np.array([[4.0]]), np.array([2.0]), np.array([[1.0]]), np.array([[1.0]]))
    assert np.allclose(m_new, [1.6]) and np.allclose(P_new, [[0.8]]), "스칼라: K = 4/5, m = 0.8*2, P = 0.2*4"


def test_huge_noise_ignores_measurement():
    m, P, y, H, _ = random_problem(3, 2, 3)
    m_new, P_new = kf_update(m, P, y, H, 1e12 * np.eye(2))
    assert np.allclose(m_new, m, atol=1e-6) and np.allclose(P_new, P, atol=1e-6), "R이 매우 크면 갱신이 거의 없어야 합니다"


def test_variance_shrinks_and_symmetric():
    m, P, y, H, R = random_problem(4, 2, 4)
    _, P_new = kf_update(m, P, y, H, R)
    assert np.allclose(P_new, P_new.T), "공분산은 대칭이어야 합니다"
    assert np.all(np.linalg.eigvalsh(P - P_new) > -1e-10), "갱신 후 공분산은 갱신 전보다 커질 수 없습니다"
    assert np.all(np.diag(H @ P_new @ H.T) < np.diag(H @ P @ H.T)), "측정한 방향의 분산은 줄어야 합니다"
