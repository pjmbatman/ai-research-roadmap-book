"""3.6 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/environment-diagnosis/cells.py
"""
# %% cell 1 — 오류 메시지로 실패한 층을 가린다
import re

rules = [  # (메시지 패턴, 실패한 층, 먼저 확인할 것)
    (r"couldn't communicate with the NVIDIA driver", "드라이버", "nvidia-smi, dkms status, uname -r"),
    (r"Driver/library version mismatch", "드라이버", "재부팅 여부, /proc/driver/nvidia/version"),
    (r"no kernel image is available", "바이너리와 GPU 아키텍처", "torch.cuda.get_arch_list()"),
    (r"cannot open shared object file", "동적 링커", "ldd, LD_LIBRARY_PATH"),
    (r"undefined symbol", "ABI·빌드 버전", "빌드한 torch 버전, C++ ABI"),
    (r"No module named", "Python 환경", "sys.executable, pip --version"),
    (r"CUDA out of memory", "자원(GPU 메모리)", "nvidia-smi 사용량, batch 크기"),
    (r"Segmentation fault", "네이티브 확장 코드", "python -X faulthandler, 최소 재현"),
]

def triage(msg):
    for pattern, layer, check in rules:
        if re.search(pattern, msg):
            return layer, check
    return "미분류", "전체 traceback과 환경 보고서"

for msg in ["RuntimeError: CUDA error: no kernel image is available for execution on the device",
            "OSError: libcudart.so.12: cannot open shared object file: No such file or directory",
            "ModuleNotFoundError: No module named 'torch'",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "Segmentation fault (core dumped)"]:
    layer, check = triage(msg)
    print(f"{msg[:48]:48s} -> [{layer}] {check}")

# %% cell 2 — 환경 보고서: 코드가 직접 수집한다
import json
import platform
import sys
from importlib import metadata

def env_report(packages):
    r = {"python": platform.python_version(), "platform": platform.platform(),
         "in_venv": sys.prefix != sys.base_prefix, "packages": {}}
    for name in packages:
        try:
            r["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            r["packages"][name] = None                 # 없는 것도 기록한다
    return r

print(json.dumps(env_report(["numpy", "matplotlib", "torch"]), indent=2))

# %% cell 3 — 변경 전후 보고서 비교
before = {"kernel": "6.8.0-45-generic", "driver": "550.54.15", "driver_max_cuda": "12.4",
          "torch": "2.5.1+cu124", "numpy": "2.1.3", "smoke_test": "pass"}
after = {"kernel": "6.8.0-48-generic", "driver": "550.54.15", "driver_max_cuda": "12.4",
         "torch": "2.5.1+cu124", "numpy": "2.2.0", "smoke_test": "fail"}

for key in before:
    if before[key] != after[key]:
        print(f"{key:12s} {before[key]:>18s} -> {after[key]}")
