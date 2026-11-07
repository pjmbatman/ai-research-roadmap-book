import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summarize = load().summarize

LOG = """2026/03/02 10:00:01.123, 0, 98, 30512, 40960
2026/03/02 10:00:01.125, 1, 0, 38200, 40960
2026/03/02 10:00:02.130, 0, 97, 30600, 40960
2026/03/02 10:00:02.131, 1, 2, 38200, 40960
2026/03/02 10:00:03.140, 0, 0, 30600, 40960
2026/03/02 10:00:03.141, 1, 1, 38200, 40960
"""


def test_keys_are_int_gpu_indices():
    s = summarize(LOG)
    assert sorted(s) == [0, 1], f"키는 GPU 번호(int) 0, 1이어야 합니다. 현재 {sorted(s)}"


def test_mean_and_max():
    s = summarize(LOG)
    assert s[0]["mean_util"] == pytest.approx((98 + 97 + 0) / 3), "사용률은 표본 평균입니다"
    assert s[0]["max_mem_mib"] == 30600, "memory.used의 최댓값입니다"
    assert s[1]["mean_util"] == pytest.approx(1.0)
    assert s[1]["max_mem_mib"] == 38200


def test_idle_holding_flag():
    s = summarize(LOG)
    assert s[1]["idle_holding"] is True, "GPU 1은 사용률 평균 1%인데 메모리 38 GB를 잡고 있습니다"
    assert s[0]["idle_holding"] is False, "GPU 0은 한 표본이 0%여도 평균 사용률이 높습니다"


def test_empty_gpu_is_not_idle_holding():
    log = "2026/03/02 10:00:01.000, 3, 0, 4, 40960\n2026/03/02 10:00:02.000, 3, 0, 4, 40960\n"
    assert summarize(log)[3]["idle_holding"] is False, "메모리를 거의 쓰지 않는 GPU는 비어 있는 것이지 유령 점유가 아닙니다"


def test_thresholds_are_parameters():
    s = summarize(LOG, idle_util_pct=50, held_mem_mib=40000)
    assert s[1]["idle_holding"] is False, "held_mem_mib=40000이면 38200 MiB는 기준을 넘지 않습니다"


def test_header_blank_and_na():
    log = ("timestamp, index, utilization.gpu [%], memory.used [MiB], memory.total [MiB]\n"
           "\n"
           "2026/03/02 10:00:01.000, 0, [N/A], 1000, 40960\n"
           "2026/03/02 10:00:02.000, 0, 40, [N/A], 40960\n"
           "2026/03/02 10:00:03.000, 0, 60, 2000, 40960\n")
    s = summarize(log)
    assert s[0]["mean_util"] == pytest.approx(50.0), "[N/A] 표본은 평균에서 뺍니다"
    assert s[0]["max_mem_mib"] == 2000


def test_all_na_util():
    log = "2026/03/02 10:00:01.000, 2, [N/A], 5000, 40960\n"
    s = summarize(log)
    assert s[2]["mean_util"] is None, "유효한 사용률 표본이 없으면 None입니다"
    assert s[2]["idle_holding"] is False
