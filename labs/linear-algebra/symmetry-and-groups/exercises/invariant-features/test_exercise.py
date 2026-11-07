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


invariant_features = load().invariant_features


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def feats(X):
    f = np.asarray(invariant_features(X), dtype=float)
    assert f.ndim == 1, f"특징은 1차원 배열이어야 합니다. 현재 shape {f.shape}"
    return f


@pytest.mark.parametrize("seed", range(5))
def test_invariant_to_rotation_translation_permutation(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 2))
    Y = (X @ rot(rng.uniform(0, 2 * np.pi)).T + rng.normal(size=2) * 5)[rng.permutation(6)]
    assert np.allclose(feats(X), feats(Y)), "회전·평행이동·순서 바꾸기를 적용해도 특징이 같아야 합니다"


def test_each_symmetry_separately():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(5, 2))
    f = feats(X)
    assert np.allclose(f, feats(X @ rot(0.8).T)), "회전에 불변이어야 합니다"
    assert np.allclose(f, feats(X + np.array([3.0, -1.0]))), "평행이동에 불변이어야 합니다"
    assert np.allclose(f, feats(X[::-1])), "순서 바꾸기에 불변이어야 합니다"


def test_fixed_length():
    rng = np.random.default_rng(11)
    assert len(feats(rng.normal(size=(7, 2)))) == len(feats(rng.normal(size=(7, 2)))), "같은 N이면 길이가 같아야 합니다"


def test_distinguishes_scaled_cloud():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(6, 2))
    f1, f2 = feats(X), feats(1.5 * X)
    assert f1.shape != f2.shape or not np.allclose(f1, f2), "크기가 1.5배인 점구름은 모양이 다르므로 구별해야 합니다"


def test_distinguishes_moved_point():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(6, 2))
    Y = X.copy()
    Y[2] += np.array([0.7, -0.4])
    f1, f2 = feats(X), feats(Y)
    assert f1.shape != f2.shape or not np.allclose(f1, f2), "점 하나를 옮겨 모양이 바뀌면 특징도 달라져야 합니다"
