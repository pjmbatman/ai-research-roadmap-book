import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).parent


def load():
    """EXERCISE_MODULE(기본값 solution)이 가리키는 같은 폴더의 모듈을 불러온다."""
    name = os.environ.get("EXERCISE_MODULE", "solution")
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summarize = load().summarize

SMI = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
"""
NVCC = """\
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 11.8, V11.8.89
"""


def test_parse_all_three():
    r = summarize(SMI, NVCC, "12.1")
    assert r["driver"] == "550.54.15", f"드라이버 버전을 읽지 못했습니다: {r['driver']}"
    assert r["driver_cuda"] == "12.4", "nvidia-smi의 CUDA Version은 드라이버가 지원하는 최대 CUDA입니다"
    assert r["toolkit"] == "11.8", "toolkit 버전은 nvcc 출력의 release 뒤 숫자입니다"
    assert r["torch_ok"] is True


def test_toolkit_not_needed():
    r = summarize(SMI, None, "12.1")
    assert r["toolkit"] is None
    assert r["torch_ok"] is True, "toolkit이 없어도 PyTorch wheel은 동작합니다(드라이버만 필요)"


def test_runtime_newer_than_driver():
    assert summarize(SMI, NVCC, "12.6")["torch_ok"] is False
    assert summarize(SMI, NVCC, "13.0")["torch_ok"] is False


def test_compare_as_numbers_not_strings():
    smi = SMI.replace("CUDA Version: 12.4 ", "CUDA Version: 12.10")
    assert summarize(smi, None, "12.4")["torch_ok"] is True, "12.4 <= 12.10 입니다. 문자열이 아니라 숫자로 비교하세요"
    assert summarize(SMI, None, "12.10")["torch_ok"] is False, "12.10 > 12.4 입니다. 문자열이 아니라 숫자로 비교하세요"


def test_no_driver():
    r = summarize(None, NVCC, "12.1")
    assert r["driver"] is None and r["driver_cuda"] is None
    assert r["torch_ok"] is False, "드라이버가 없으면 CUDA용 PyTorch는 GPU를 쓸 수 없습니다"


def test_cpu_build():
    assert summarize(SMI, NVCC, None)["torch_ok"] is None, "CPU 빌드(torch.version.cuda가 None)는 판정 대상이 아닙니다"
