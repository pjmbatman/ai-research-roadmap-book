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


group_kfold = load().group_kfold


def check(groups, k):
    fold = np.asarray(group_kfold(groups, k))
    groups = np.asarray(groups)
    assert fold.shape == groups.shape, "fold 배열의 길이는 샘플 수와 같아야 합니다"
    assert set(np.unique(fold)) == set(range(k)), f"fold 번호는 0..{k - 1}를 모두 써야 합니다"
    for g in np.unique(groups):
        assert len(np.unique(fold[groups == g])) == 1, f"그룹 {g}가 여러 fold에 나뉘었습니다"
    sizes = np.bincount(fold, minlength=k)
    biggest = np.unique(groups, return_counts=True)[1].max()
    assert sizes.max() - sizes.min() <= biggest, f"fold 크기가 고르지 않습니다: {sizes.tolist()}"
    return fold


@pytest.mark.parametrize("k", [2, 3, 5])
def test_random_group_sizes(k):
    rng = np.random.default_rng(k)
    sizes = rng.integers(1, 40, size=23)                      # 크기가 제각각인 그룹 23개
    groups = np.repeat(np.arange(23), sizes)
    rng.shuffle(groups)                                       # 샘플 순서는 섞여 있다
    check(groups, k)


def test_string_groups():
    groups = ["cellA"] * 5 + ["cellB"] * 3 + ["cellC"] * 4 + ["cellD"] * 4
    check(groups, 2)


def test_one_huge_group():
    groups = np.array([0] * 50 + [1] * 5 + [2] * 5 + [3] * 5)
    fold = check(groups, 2)
    assert np.sum(fold == fold[0]) == 50, "큰 그룹 하나가 fold 하나를 차지하고 나머지가 다른 fold에 가야 합니다"


def test_too_few_groups():
    with pytest.raises(ValueError):
        group_kfold([0, 0, 1, 1], 3)
