"""6.6 실습: pipeline.py의 연구 불변조건 테스트.

실행(저장소 루트에서):
  cd labs && uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
  cd labs && INJECT_BUG=leak uv run pytest -q research-code-engineering/testing-and-debugging/test_pipeline.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("rce_pipeline", Path(__file__).parent / "pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


@pytest.fixture
def groups():
    """시료 30개 × 반복 측정 4회. (120,)"""
    return np.repeat(np.arange(30), 4)


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 3))                     # (40, 3)
    w_true = np.array([1.0, -2.0, 0.5])              # (3,)
    return A, A @ w_true, w_true


# --- 데이터 분할 ---------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_is_a_partition(groups, seed):
    tr, te = pipeline.split_by_group(groups, 0.2, seed)
    assert len(np.intersect1d(tr, te)) == 0, "같은 행이 train과 test에 모두 있다"
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(len(groups))), "빠진 행이 있다"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_has_no_group_leakage(groups, seed):
    tr, te = pipeline.split_by_group(groups, 0.2, seed)
    shared = np.intersect1d(groups[tr], groups[te])
    assert len(shared) == 0, f"같은 시료 {len(shared)}개가 train과 test에 나뉘어 있다: {shared[:5]}"


def test_split_is_deterministic(groups):
    a = pipeline.split_by_group(groups, 0.2, seed=7)
    b = pipeline.split_by_group(groups, 0.2, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --- metric 집계 ---------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 7, 32, 100, 1000])
def test_mae_does_not_depend_on_batch_size(batch_size):
    rng = np.random.default_rng(1)
    y, y_hat = rng.normal(size=100), rng.normal(size=100)          # (100,), (100,)
    expected = np.mean(np.abs(y - y_hat))                           # 독립적인 기준: 한 번에 계산
    np.testing.assert_allclose(pipeline.mae(y, y_hat, batch_size), expected, rtol=1e-12)


def test_mae_known_value():
    y, y_hat = np.array([0.0, 0.0, 0.0]), np.array([1.0, -2.0, 3.0])
    assert pipeline.mae(y, y_hat, batch_size=2) == pytest.approx(2.0)


# --- checkpoint 복원 -----------------------------------------------------

def test_resume_from_checkpoint_is_identical(problem, tmp_path):
    A, b, _ = problem
    straight = pipeline.Trainer(A, b)
    straight.train(20)

    first = pipeline.Trainer(A, b)
    first.train(10)
    first.save(tmp_path / "ckpt.npz")
    resumed = pipeline.Trainer(A, b)
    resumed.load(tmp_path / "ckpt.npz")
    resumed.train(10)

    assert resumed.step == straight.step == 20
    np.testing.assert_allclose(resumed.w, straight.w, rtol=1e-12, atol=1e-12,
                               err_msg="중간에 저장·복원한 학습이 이어서 학습한 결과와 다르다")


# --- smoke test: 작은 데이터에 과적합할 수 있는가 -------------------------

def test_overfits_tiny_problem(problem):
    A, b, w_true = problem
    t = pipeline.Trainer(A, b)
    t.train(500)
    assert t.loss() < 1e-10
    np.testing.assert_allclose(t.w, w_true, atol=1e-6)
