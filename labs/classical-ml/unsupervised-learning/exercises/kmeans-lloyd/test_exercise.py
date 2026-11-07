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


kmeans = load().kmeans


def test_hand_example():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    C, labels, history = kmeans(X, np.array([[0.0], [12.0]]))
    assert np.allclose(C, [[1.0], [11.0]]), f"중심은 [[1], [11]]이어야 합니다. 현재 {np.asarray(C).tolist()}"
    assert list(labels) == [0, 0, 1, 1], f"배정은 [0, 0, 1, 1]이어야 합니다. 현재 {list(labels)}"
    assert np.isclose(history[0], 8.0), "첫 배정 직후 inertia는 0 + 4 + 4 + 0 = 8입니다"
    assert np.isclose(history[-1], 4.0), "수렴했을 때 inertia는 1 + 1 + 1 + 1 = 4입니다"


def test_init_not_modified():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    init = np.array([[0.0], [12.0]])
    kmeans(X, init)
    assert np.allclose(init, [[0.0], [12.0]]), "init_centers 배열을 직접 고치면 안 됩니다(복사본을 쓰세요)"


def test_history_non_increasing_and_consistent():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (60, 2)), rng.normal(5, 1, (60, 2)), rng.normal([0, 6], 1, (60, 2))]   # (180, 2)
    C, labels, history = kmeans(X, X[[0, 1, 2]])
    h = np.asarray(history)
    assert np.all(np.diff(h) <= 1e-9), "Lloyd 반복에서 inertia는 증가하지 않아야 합니다"
    nearest = ((X[:, None, :] - np.asarray(C)[None]) ** 2).sum(-1).argmin(1)
    assert np.array_equal(labels, nearest), "반환한 labels는 반환한 centers에 대한 최근접 배정이어야 합니다"
    assert np.shape(C) == (3, 2) and np.shape(labels) == (180,)


def test_bad_init_gets_stuck():
    X = np.array([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]])
    _, _, good = kmeans(X, np.array([[0.0], [10.0], [20.0]]))
    _, labels, bad = kmeans(X, np.array([[0.0], [1.0], [15.0]]))
    assert np.isclose(good[-1], 1.5), "좋은 초기값에서는 inertia 1.5에 수렴합니다"
    assert np.isclose(bad[-1], 101.0), "나쁜 초기값 [0, 1, 15]에서는 inertia 101인 국소해에 멈춥니다"
    assert list(labels) == [0, 1, 2, 2, 2, 2]


def test_empty_cluster_kept():
    X = np.array([[0.0], [1.0]])
    C, labels, _ = kmeans(X, np.array([[0.5], [100.0]]))
    assert np.isclose(np.asarray(C)[1, 0], 100.0), "점이 배정되지 않은 중심은 그대로 두어야 합니다"
    assert list(labels) == [0, 0]
