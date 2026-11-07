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


pick_gpus = load().pick_gpus

GPUS = """0, GPU-aaaa, 30512, 40960
1, GPU-bbbb, 38200, 40960
2, GPU-cccc, 4, 40960
3, GPU-dddd, 4, 40960
"""
APPS = """GPU-aaaa, 5310, 30500
GPU-bbbb, 4190, 38190
"""


def test_problem_example():
    assert pick_gpus(GPUS, APPS, 2, 20000) == "2,3"


def test_returns_lowest_indices():
    assert pick_gpus(GPUS, APPS, 1, 20000) == "2", "비어 있는 GPU 중 번호가 작은 것부터 고릅니다"


def test_process_by_uuid_not_index():
    # 프로세스가 GPU-cccc(번호 2)에 있다. uuid를 번호로 착각하면 틀린다.
    apps = "GPU-cccc, 777, 100\n"
    gpus = "0, GPU-cccc, 120, 40960\n1, GPU-aaaa, 4, 40960\n2, GPU-bbbb, 4, 40960\n"
    assert pick_gpus(gpus, apps, 2, 20000) == "1,2", "프로세스 목록의 uuid로 GPU를 찾아야 합니다"


def test_memory_without_visible_process_is_not_free():
    gpus = "0, GPU-aaaa, 35000, 40960\n1, GPU-bbbb, 4, 40960\n"
    assert pick_gpus(gpus, "", 1, 20000) == "1", "프로세스가 안 보여도 메모리가 차 있으면 비어 있지 않습니다"


def test_small_process_still_occupies():
    apps = "GPU-dddd, 900, 300\n"
    assert pick_gpus(GPUS, APPS + apps, 1, 20000) == "2", "메모리를 조금만 써도 프로세스가 있으면 남의 GPU입니다"


def test_free_memory_threshold_inclusive():
    gpus = "0, GPU-aaaa, 20960, 40960\n"
    assert pick_gpus(gpus, "", 1, 20000) == "0", "빈 메모리가 min_free_mib와 같으면 충분합니다"


def test_not_enough_raises():
    with pytest.raises(RuntimeError) as e:
        pick_gpus(GPUS, APPS, 3, 20000)
    assert not isinstance(e.value, NotImplementedError), "함수를 완성한 뒤, 비어 있는 GPU가 모자라면 RuntimeError를 내야 합니다"


def test_blank_lines_ignored():
    assert pick_gpus("\n" + GPUS + "\n", "\n" + APPS + "\n\n", 2, 20000) == "2,3"
