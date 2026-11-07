import importlib.util
import math
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


summarize = load().summarize


def run(rid, lr, seed, status, loss, wd=0.0):
    return {"run_id": rid, "config": {"lr": lr, "seed": seed, "wd": wd}, "status": status,
            "metrics": {"val_loss": loss}}


RUNS = [
    run("r1", 0.01, 0, "finished", 0.50), run("r2", 0.01, 1, "finished", 0.54), run("r3", 0.01, 2, "finished", 0.58),
    run("r4", 0.1, 0, "failed", 7.9), run("r5", 0.1, 1, "finished", 0.31), run("r6", 0.1, 2, "finished", 0.29),
    run("r7", 0.3, 0, "failed", 9.1), run("r8", 0.3, 1, "running", 0.8),
]


def test_mean_and_std_of_finished_runs():
    t = summarize(RUNS, ["lr"], "val_loss")
    assert np.isclose(t[(0.01,)]["mean"], 0.54)
    assert np.isclose(t[(0.01,)]["std"], np.std([0.50, 0.54, 0.58], ddof=1)), "표준편차는 표본 표준편차(ddof=1)입니다"


def test_unfinished_runs_are_excluded_but_counted():
    t = summarize(RUNS, ["lr"], "val_loss")
    assert np.isclose(t[(0.1,)]["mean"], 0.30), "실패한 run의 마지막 손실(7.9)이 평균에 섞였습니다"
    assert t[(0.1,)]["n"] == 2 and t[(0.1,)]["missing"] == 1, "끝나지 않은 run은 평균에서 빼고 missing으로 셉니다"
    assert t[(0.1,)]["run_ids"] == ["r5", "r6"]


def test_group_with_no_finished_run_is_kept():
    t = summarize(RUNS, ["lr"], "val_loss")
    assert (0.3,) in t, "모든 run이 실패한 그룹도 표에 남겨야 실패가 드러납니다"
    g = t[(0.3,)]
    assert g["n"] == 0 and g["missing"] == 2, "끝난 run 0개, 빠진 run 2개여야 합니다"
    assert math.isnan(g["mean"]) and math.isnan(g["std"]), "끝난 run이 없으면 평균과 표준편차는 nan입니다"


def test_single_run_has_nan_std():
    t = summarize([run("a", 0.1, 0, "finished", 0.3)], ["lr"], "val_loss")
    assert t[(0.1,)]["n"] == 1 and math.isnan(t[(0.1,)]["std"]), "run 하나로는 표본 표준편차를 정의할 수 없습니다(nan)"


def test_multiple_group_keys():
    runs = [run("a", 0.1, 0, "finished", 0.3, wd=0.0), run("b", 0.1, 1, "finished", 0.5, wd=0.1),
            run("c", 0.1, 2, "finished", 0.4, wd=0.1)]
    t = summarize(runs, ["lr", "wd"], "val_loss")
    assert set(t) == {(0.1, 0.0), (0.1, 0.1)}, "그룹 키는 group_by 순서대로 값을 모은 tuple입니다"
    assert np.isclose(t[(0.1, 0.1)]["mean"], 0.45) and t[(0.1, 0.1)]["run_ids"] == ["b", "c"]
