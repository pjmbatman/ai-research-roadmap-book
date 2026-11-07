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


check_column = load().check_column
TEMP = {"min": -20.0, "max": 60.0, "missing_codes": [-999.0]}


def test_counts_basic():
    v = np.array([25.1, -999.0, np.nan, 298.4, 61.0, -20.0, 60.0])
    r = check_column(v, TEMP)
    assert r == {"missing": 2, "out_of_range": 2, "valid": 3}, f"결과가 다릅니다: {r}"


def test_sentinel_is_not_out_of_range():
    r = check_column(np.array([-999.0, -999.0, 20.0]), TEMP)
    assert r["out_of_range"] == 0, "결측 코드 -999는 범위 밖 값이 아니라 결측으로 세야 합니다"
    assert r["missing"] == 2


def test_bounds_inclusive():
    r = check_column(np.array([-20.0, 60.0]), TEMP)
    assert r["valid"] == 2, "허용 범위의 양 끝 값은 유효합니다"


def test_without_missing_codes():
    spec = {"min": 2.5, "max": 4.25}
    r = check_column(np.array([3.7, np.nan, 3702.0]), spec)
    assert r == {"missing": 1, "out_of_range": 1, "valid": 1}, f"missing_codes가 없을 때의 결과가 다릅니다: {r}"


def test_counts_sum_to_length():
    rng = np.random.default_rng(0)
    v = rng.normal(20.0, 30.0, size=500)
    v[rng.choice(500, 40, replace=False)] = -999.0
    v[rng.choice(500, 30, replace=False)] = np.nan
    r = check_column(v, TEMP)
    assert sum(r.values()) == 500, "세 개수의 합은 전체 값의 수와 같아야 합니다"
    assert all(isinstance(x, int) for x in r.values()), "개수는 int로 반환합니다"
