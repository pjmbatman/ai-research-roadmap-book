"""3.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/software-stack/cells.py
"""
# %% cell 1 — 세 가지 'CUDA 버전' 읽기
import re

smi = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |"
nvcc = "Cuda compilation tools, release 12.1, V12.1.105"
torch_cuda = "12.1"                      # torch.version.cuda가 돌려주는 값의 예

driver = re.search(r"Driver Version:\s*([\d.]+)", smi).group(1)
driver_cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi).group(1)
toolkit = re.search(r"release\s+([\d.]+)", nvcc).group(1)

print("드라이버 버전                :", driver)
print("드라이버가 지원하는 최대 CUDA :", driver_cuda, " <- nvidia-smi")
print("설치된 CUDA toolkit          :", toolkit, " <- nvcc --version")
print("PyTorch가 빌드된 CUDA        :", torch_cuda, " <- torch.version.cuda")

# %% cell 2 — 드라이버와 CUDA runtime의 호환성 판정
def ver(s):
    return tuple(int(p) for p in s.split("."))   # "12.4" -> (12, 4)

def judge(driver_cuda, runtime_cuda):
    d, r = ver(driver_cuda), ver(runtime_cuda)
    if r <= d:
        return "실행 가능"
    if r[0] == d[0]:
        return "같은 major: minor version compatibility (일부 기능 제한)"
    return "실행 불가: 드라이버 업그레이드 필요"

for runtime in ["11.8", "12.1", "12.4", "12.6", "13.0"]:
    print(f"드라이버 최대 CUDA 12.4, runtime {runtime:>4} -> {judge('12.4', runtime)}")

# %% cell 3 — 이 컴퓨터의 계층별 버전 수집
import platform
import shutil
import subprocess

def first_line(cmd, last=False):
    if shutil.which(cmd[0]) is None:
        return f"{cmd[0]} 없음"
    lines = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
    return (lines[-1] if last else lines[0]) if lines else "(출력 없음)"

print("CPU 아키텍처 :", platform.machine())
print("kernel       :", platform.release())
print("C 라이브러리  :", " ".join(platform.libc_ver()))
print("드라이버      :", first_line(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]))
print("toolkit      :", first_line(["nvcc", "--version"], last=True))
print("gcc          :", first_line(["gcc", "--version"]))
print("Python       :", platform.python_version())
try:
    import torch
    print("PyTorch      :", torch.__version__, "/ CUDA", torch.version.cuda)
except ImportError:
    print("PyTorch      : 설치되지 않음")
