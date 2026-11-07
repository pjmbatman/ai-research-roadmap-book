"""3.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/ubuntu-and-cuda/cells.py
"""
# %% cell 1 — LTS 지원 기간과 프로젝트 기간
from datetime import date

lts_release = {"20.04": date(2020, 4, 1), "22.04": date(2022, 4, 1), "24.04": date(2024, 4, 1)}
project_start, project_end = date(2026, 11, 1), date(2029, 12, 31)

for name, released in lts_release.items():
    eol = released.replace(year=released.year + 5)      # 표준 지원 약 5년(근사)
    if eol <= project_start:
        status = "이미 표준 지원 종료"
    elif eol < project_end:
        months = (eol.year - project_start.year) * 12 + eol.month - project_start.month
        status = f"프로젝트 시작 후 {months}개월에 종료"
    else:
        status = "프로젝트 기간 전체를 덮음"
    print(f"Ubuntu {name}: 표준 지원 ~{eol:%Y-%m}  -> {status}")

# %% cell 2 — 여러 CUDA가 있을 때 어느 nvcc가 실행되는가
files = {"/usr/local/cuda-12.1/bin/nvcc", "/usr/local/cuda-12.4/bin/nvcc", "/usr/bin/gcc"}
links = {"/usr/local/cuda": "/usr/local/cuda-12.4"}    # 심볼릭 링크

def realpath(p):                                       # readlink -f 흉내
    for src, dst in links.items():
        if p.startswith(src + "/"):
            return dst + p[len(src):]
    return p

def which(name, path_var):
    for d in path_var.split(":"):                      # 앞에서부터, 처음 찾은 것
        candidate = f"{d.rstrip('/')}/{name}"
        if realpath(candidate) in files:
            return candidate
    return None

for path_var in ["/usr/local/cuda/bin:/usr/bin",
                 "/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/usr/bin",
                 "/usr/bin"]:
    found = which("nvcc", path_var)
    print(f"PATH={path_var}\n  -> {found}  (실제 파일: {realpath(found) if found else '-'})")

# %% cell 3 — 드라이버 최소 버전 확인: 문자열 비교의 함정
min_driver = {"12.0 GA": "525.60.13", "12.6 Update 3": "560.35.05", "12.8 Update 1": "570.124.06"}
installed = "525.105.17"

def vtuple(s):
    return tuple(int(p) for p in s.split("."))   # "525.105.17" -> (525, 105, 17)

for cuda, need in min_driver.items():
    ok_str = installed >= need                       # 잘못된 방법
    ok_num = vtuple(installed) >= vtuple(need)      # 올바른 방법
    print(f"CUDA {cuda:13s} 최소 {need:>10s}  문자열 비교: {ok_str!s:5s}  숫자 비교: {ok_num}")
