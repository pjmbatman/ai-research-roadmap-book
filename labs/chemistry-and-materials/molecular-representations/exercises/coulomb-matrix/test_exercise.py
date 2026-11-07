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


coulomb_eigs = load().coulomb_eigs

Z_W = np.array([8.0, 1.0, 1.0])
R_W = np.array([[0.0, 0.0, 0.0], [1.431, 1.108, 0.0], [-1.431, 1.108, 0.0]])   # 물 (bohr)


def random_rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.linalg.det(Q))


def test_hydrogen_molecule():
    e = coulomb_eigs(np.array([1.0, 1.0]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    assert np.shape(e) == (2,), f"고윳값 배열의 shape은 (2,)여야 합니다. 현재 {np.shape(e)}"
    assert np.allclose(e, [0.5 + 1 / 1.4, 0.5 - 1 / 1.4]), "H2의 고윳값은 0.5 ± 1/1.4입니다. 대각은 Z^2.4/2, 비대각은 Z_i Z_j / 거리"


def test_sorted_descending():
    e = coulomb_eigs(Z_W, R_W)
    assert np.all(np.diff(e) <= 0), "고윳값을 큰 것부터 정렬해야 합니다"
    assert np.isclose(e.sum(), 0.5 * (8 ** 2.4 + 2)), "고윳값의 합은 대각합 sum(Z^2.4)/2와 같아야 합니다"


def test_rotation_translation_invariance():
    rng = np.random.default_rng(0)
    for _ in range(3):
        R2 = R_W @ random_rotation(rng).T + rng.normal(size=3) * 5
        assert np.allclose(coulomb_eigs(Z_W, R2), coulomb_eigs(Z_W, R_W)), "회전·평행이동한 분자에서 결과가 달라졌습니다"


def test_permutation_invariance():
    perm = np.array([1, 2, 0])
    assert np.allclose(coulomb_eigs(Z_W[perm], R_W[perm]), coulomb_eigs(Z_W, R_W)), "원자 번호를 바꿔 매겼을 때 결과가 달라졌습니다"
