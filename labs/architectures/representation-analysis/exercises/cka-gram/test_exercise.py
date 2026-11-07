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


cka_gram = load().cka_gram


def feature_cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))


def test_matches_feature_form():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 6))
    B = np.tanh(A @ rng.normal(size=(6, 9))) + 0.3 * rng.normal(size=(40, 9))
    assert np.isclose(cka_gram(A @ A.T, B @ B.T), feature_cka(A, B)), \
        "선형 Gram 행렬에서는 특징 공간 공식과 같아야 합니다(Gram 행렬을 중심화했는지 확인)"


def test_self_similarity_is_one():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 5))
    assert np.isclose(cka_gram(A @ A.T, A @ A.T), 1.0), "자기 자신과의 CKA는 1이어야 합니다"


def test_orthogonal_and_scale_invariance():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    base = cka_gram(A @ A.T, B @ B.T)
    AQ = 5.0 * A @ Q
    assert np.isclose(cka_gram(AQ @ AQ.T, B @ B.T), base), "직교 변환과 등방 스케일에 불변이어야 합니다"


def test_different_dimensions():
    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(25, 3)), rng.normal(size=(25, 40))
    v = cka_gram(A @ A.T, B @ B.T)
    assert 0.0 <= v <= 1.0 and np.isclose(v, feature_cka(A, B)), "차원이 달라도 0과 1 사이의 같은 값이어야 합니다"


def test_offset_does_not_matter():
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    A2 = A + 100.0                                      # 모든 표본에 같은 벡터를 더함
    assert np.isclose(cka_gram(A2 @ A2.T, B @ B.T), cka_gram(A @ A.T, B @ B.T)), \
        "중심화했다면 모든 표본에 같은 벡터를 더해도 값이 같아야 합니다"


def test_returns_float():
    A = np.eye(4)[:, :2]
    assert np.ndim(cka_gram(A @ A.T + np.eye(4), np.eye(4) * 2)) == 0, "float 하나를 반환해야 합니다"
