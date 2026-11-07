"""3.1 실습: 서버의 소프트웨어 스택을 계층별로 수집하고 호환성을 판정한다.

실행: uv run --project labs python labs/environments/software-stack/collect_versions.py
      (PyTorch가 있는 환경에서 실행하면 PyTorch·cuDNN·NCCL 계층까지 채워진다)

표준 라이브러리만 쓴다. 없는 도구는 '없음'으로 기록하고 멈추지 않는다.
"""
import platform
import re
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    """명령을 실행해 stdout 전체를 돌려준다. 명령이 없거나 실패하면 None."""
    if shutil.which(cmd[0]) is None:
        return None
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s))


def os_release():
    f = Path("/etc/os-release")
    if not f.exists():
        return None
    kv = dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)
    return kv.get("PRETTY_NAME", "").strip('"')


def collect():
    r = {}
    r["cpu_arch"] = platform.machine()
    r["os"] = os_release()
    r["kernel"] = platform.release()
    r["glibc"] = " ".join(platform.libc_ver()) or None
    # 커널 모듈로 올라간 드라이버 (사용자 공간 라이브러리와 다를 수 있다)
    proc = Path("/proc/driver/nvidia/version")
    m = re.search(r"Kernel Module\s+([\d.]+)", proc.read_text()) if proc.exists() else None
    r["driver_kernel_module"] = m.group(1) if m else None
    smi = run(["nvidia-smi"])
    r["driver"] = re.search(r"Driver Version:\s*([\d.]+)", smi).group(1) if smi else None
    r["driver_max_cuda"] = re.search(r"CUDA Version:\s*([\d.]+)", smi).group(1) if smi else None
    gpus = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    r["gpus"] = gpus.splitlines() if gpus else []
    nvcc = run(["nvcc", "--version"])
    r["toolkit_nvcc"] = re.search(r"release\s+([\d.]+)", nvcc).group(1) if nvcc else None
    gcc = run(["gcc", "-dumpfullversion"])
    r["gcc"] = gcc
    r["python"] = platform.python_version()
    try:
        import torch

        r["torch"] = torch.__version__
        r["torch_cuda"] = torch.version.cuda
        r["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        r["torch_cxx11_abi"] = torch._C._GLIBCXX_USE_CXX11_ABI
        r["cuda_available"] = torch.cuda.is_available()
        r["torch_arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    except ImportError:
        r["torch"] = None
    return r


def judge(r):
    """수집한 값으로 계층 사이의 호환성을 판정한다. (판정, 근거) 목록."""
    out = []
    if r["driver"] and r["driver_kernel_module"] and r["driver"] != r["driver_kernel_module"]:
        out.append(("경고", f"커널 모듈 {r['driver_kernel_module']} != 사용자 공간 드라이버 {r['driver']}: 재부팅 필요 가능성"))
    if r.get("torch") and r.get("torch_cuda"):
        if not r["driver_max_cuda"]:
            out.append(("실패", "CUDA용 PyTorch인데 드라이버(nvidia-smi)가 없다"))
        elif ver(r["torch_cuda"])[:2] <= ver(r["driver_max_cuda"])[:2]:
            out.append(("통과", f"PyTorch CUDA {r['torch_cuda']} <= 드라이버 최대 CUDA {r['driver_max_cuda']}"))
        elif ver(r["torch_cuda"])[0] == ver(r["driver_max_cuda"])[0]:
            out.append(("주의", "같은 major, 더 높은 minor: minor version compatibility에 의존"))
        else:
            out.append(("실패", "PyTorch의 CUDA major가 드라이버보다 높다: 드라이버 업그레이드 필요"))
    elif r.get("torch"):
        out.append(("정보", "CPU 전용 PyTorch 빌드"))
    if r.get("torch_cuda") and r["toolkit_nvcc"] and ver(r["toolkit_nvcc"])[0] != ver(r["torch_cuda"])[0]:
        out.append(("주의", f"nvcc {r['toolkit_nvcc']}와 PyTorch CUDA {r['torch_cuda']}의 major가 달라 extension 빌드가 실패한다"))
    if not r["toolkit_nvcc"]:
        out.append(("정보", "CUDA toolkit(nvcc) 없음: 학습·추론에는 보통 필요 없고, extension 빌드에만 필요"))
    return out


if __name__ == "__main__":
    report = collect()
    for k, v in report.items():
        print(f"{k:22s} {v if v not in (None, []) else '-'}")
    print()
    for level, reason in judge(report):
        print(f"[{level}] {reason}")
