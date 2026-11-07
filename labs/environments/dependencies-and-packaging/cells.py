"""3.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/dependencies-and-packaging/cells.py
"""
# %% cell 1 — 직접 의존성, 전이 의존성, 겹치는 제약
meta = {   # 가상의 메타데이터: 패키지 -> {의존 패키지: 버전 제약}
    "my-analysis":     {"pandas": ">=2.2", "scipy": ">=1.11", "matplotlib": ">=3.8"},
    "pandas":          {"numpy": ">=1.26", "python-dateutil": ">=2.8"},
    "scipy":           {"numpy": ">=1.23,<2.6"},
    "matplotlib":      {"numpy": ">=1.23", "pillow": ">=8"},
    "python-dateutil": {"six": ">=1.5"},
}

direct = set(meta["my-analysis"])
seen, stack, on_numpy = set(), list(direct), []
while stack:                                   # 의존성 그래프를 따라 내려간다
    pkg = stack.pop()
    if pkg in seen:
        continue
    seen.add(pkg)
    for dep, spec in meta.get(pkg, {}).items():
        stack.append(dep)
        if dep == "numpy":
            on_numpy.append(f"{pkg}: numpy{spec}")

print("직접 의존성:", sorted(direct))
print("전이 의존성:", sorted(seen - direct))
print("numpy에 걸린 제약:", sorted(on_numpy))

# %% cell 2 — wheel 파일 이름의 platform tag 읽기
import re

files = ["numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
         "numpy-2.5.4-cp313-cp313-musllinux_1_2_x86_64.whl",
         "numpy-2.5.4-cp313-cp313-win_amd64.whl",
         "torch-2.15.0-cp313-cp313-manylinux_2_28_x86_64.whl",
         "numpy-2.5.4.tar.gz"]

def check(fn, py, glibc, arch="x86_64"):
    if not fn.endswith(".whl"):
        return "sdist: 소스에서 빌드"
    name, version, pytag, abi, plats = fn[:-4].split("-")     # build tag가 없는 이름
    for plat in plats.split("."):
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", plat)
        if pytag == py and m and (int(m[1]), int(m[2])) <= glibc and m[3] == arch:
            return f"설치 가능 ({plat})"
    return "해당 없음"

for glibc in [(2, 39), (2, 17)]:
    print(f"== Linux x86_64, CPython 3.13, glibc {glibc[0]}.{glibc[1]}")
    for fn in files:
        print(f"  {fn:72s} {check(fn, 'cp313', glibc)}")

# %% cell 3 — 작은 resolver: 높은 버전부터 시도하고 막히면 되돌아간다
def V(s):
    return tuple(int(p) for p in s.split("."))

def ok(v, spec):                                 # 이 예에서 쓰는 >=, ==, < 만 지원
    for c in spec.split(","):
        op = c[:2] if c[:2] in (">=", "==") else c[0]
        a, b = V(v), V(c[len(op):])
        if not {">=": a >= b, "==": a == b, "<": a < b}[op]:
            return False
    return True

available = {"pandas": ["3.1.1", "2.2.3"], "scipy": ["1.16.0", "1.13.1"], "numpy": ["2.5.4", "1.26.4"]}
deps = {("pandas", "3.1.1"): {"numpy": ">=2.0.2"}, ("pandas", "2.2.3"): {"numpy": ">=1.26"},
        ("scipy", "1.16.0"): {"numpy": ">=2.0"}, ("scipy", "1.13.1"): {"numpy": ">=1.22,<2.3"}}

def resolve(reqs, pinned):
    for name, spec in reqs:
        if name in pinned:
            if not ok(pinned[name], spec):
                return None                      # 이미 고른 버전과 충돌: 되돌아간다
            continue
        for v in available[name]:                # 높은 버전부터
            if ok(v, spec):
                found = resolve(reqs + list(deps.get((name, v), {}).items()), {**pinned, name: v})
                if found:
                    return found
        return None
    return pinned

print(resolve([("pandas", ">=2.2"), ("scipy", ">=1.11")], {}))
print(resolve([("pandas", ">=2.2"), ("scipy", ">=1.11"), ("numpy", "<2")], {}))
print(resolve([("pandas", ">=3"), ("numpy", "<2")], {}))
