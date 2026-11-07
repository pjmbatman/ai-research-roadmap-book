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


pareto_mask = load().pareto_mask


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def test_small_example():
    F = np.array([[0.9, 0.2],      # 수율 높음, 순도 낮음 -> front
                  [0.5, 0.5],      # 균형 -> front
                  [0.2, 0.9],      # front
                  [0.4, 0.4],      # (0.5, 0.5)에 지배
                  [0.9, 0.1]])     # (0.9, 0.2)에 지배
    m = pareto_mask(F)
    assert np.shape(m) == (5,), f"길이 N = 5의 배열이어야 합니다. 현재 shape {np.shape(m)}"
    assert m.tolist() == [True, True, True, False, False], f"현재 {m.tolist()}"


def test_duplicates_kept():
    F = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.1]])
    assert pareto_mask(F).tolist() == [True, True, False], "똑같은 두 점은 서로를 지배하지 않으므로 둘 다 남아야 합니다"


def test_single_objective():
    F = np.array([[0.3], [0.8], [0.5], [0.8]])
    assert pareto_mask(F).tolist() == [False, True, False, True], "목적이 하나면 최댓값(동률 포함)만 남습니다"


def test_definition_on_random_points():
    rng = np.random.default_rng(0)
    F = rng.uniform(size=(80, 3))
    m = np.asarray(pareto_mask(F), dtype=bool)
    front, rest = F[m], F[~m]
    assert m.any(), "front는 비어 있을 수 없습니다"
    for a in front:
        assert not any(dominates(b, a) for b in F), "front의 점이 다른 점에 지배됩니다"
    for b in rest:
        assert any(dominates(a, b) for a in front), "front 밖의 점은 front의 어떤 점에 지배되어야 합니다"
