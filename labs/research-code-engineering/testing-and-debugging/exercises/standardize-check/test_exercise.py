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


check_standardize = load().check_standardize


# --- 올바른 구현 두 개: 계산 순서가 달라 마지막 자릿수가 다를 수 있다 -----------

def correct_divide(x):
    mu, sd = x.mean(axis=0), x.std(axis=0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def correct_reciprocal(x):
    z = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):                       # 열마다, 역수를 곱해서
        col = x[:, j]
        mu = col.sum() / len(col)
        sd = np.sqrt(((col - mu) ** 2).sum() / len(col))
        z[:, j] = (col - mu) * (1.0 / sd) if sd > 0 else 0.0
    return z


# --- 버그가 있는 구현: 연구 코드에서 흔한 실수 ---------------------------------

def bug_wrong_axis(x):                               # 열이 아니라 행마다 표준화
    mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def bug_no_centering(x):                             # 평균을 빼지 않음
    sd = x.std(axis=0)
    return x / np.where(sd > 0, sd, 1.0)


def bug_constant_column_nan(x):                      # 표준편차 0인 열을 처리하지 않음
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - x.mean(axis=0)) / x.std(axis=0)


def bug_global_stats(x):                             # 전체 배열의 평균·표준편차 하나로 표준화
    return (x - x.mean()) / x.std()


def bug_in_place(x):                                 # 입력 배열을 덮어씀
    sd = x.std(axis=0)
    x -= x.mean(axis=0)
    x /= np.where(sd > 0, sd, 1.0)
    return x


def bug_sample_std(x):                               # ddof=1 (명세는 ddof=0)
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def passes(impl):
    try:
        check_standardize(impl)
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize("impl", [correct_divide, correct_reciprocal])
def test_correct_implementations_pass(impl):
    try:
        check_standardize(impl)
    except AssertionError as e:
        raise AssertionError(
            f"올바른 구현 {impl.__name__}이 테스트에서 떨어졌습니다. "
            f"부동소수점 비교에 허용오차를 썼는지, 0과 비교할 때 atol을 줬는지 확인하세요.\n{e}"
        ) from e


@pytest.mark.parametrize("impl", [bug_wrong_axis, bug_no_centering, bug_constant_column_nan,
                                  bug_global_stats, bug_in_place, bug_sample_std])
def test_buggy_implementations_fail(impl):
    try:
        caught = not passes(impl)
    except Exception:                                # 버그 때문에 다른 예외가 나도 잡은 것으로 본다
        caught = True
    assert caught, f"버그가 있는 구현 {impl.__name__}이 테스트를 통과했습니다. 명세의 어느 문장을 어기는지 생각해 보세요."
