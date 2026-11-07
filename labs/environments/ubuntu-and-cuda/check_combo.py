"""3.2 실습: 연구실 표준 OS·드라이버·CUDA 조합 기록을 검사한다.

실행: uv run --project labs python labs/environments/ubuntu-and-cuda/check_combo.py [combo.toml]
      인자가 없으면 같은 폴더의 combo.example.toml을 검사한다.

검사하는 것은 '조합이 기록 안에서 서로 맞는가'와 '모든 값에 근거(source·checked)가 있는가'다.
값이 공식 문서와 맞는지는 사람이 확인한다. 이 스크립트는 그 확인을 빠뜨리지 않게 돕는다.
"""
import sys
import tomllib
from pathlib import Path


def v(s):
    return tuple(int(p) for p in str(s).split("."))


def check(c):
    results = []

    def add(ok, msg):
        results.append(("통과" if ok is True else "실패" if ok is False else "확인 필요", msg))

    for section in ("os", "gpu", "driver", "toolkit", "compiler", "framework"):
        s = c.get(section, {})
        if not s.get("source") or not s.get("checked"):
            add(None, f"[{section}] 근거 없음: source와 checked(확인 날짜)를 채운다")

    os_, drv, tk, cc, fw = c["os"], c["driver"], c["toolkit"], c["compiler"], c["framework"]
    add(v(os_["standard_support_until"].replace("-", ".")) >= v(c["project_end"].replace("-", ".")),
        f"OS 표준 지원({os_['standard_support_until']})이 프로젝트 종료({c['project_end']})까지 이어지는가")
    add(v(drv["version"]) >= v(tk["min_driver"]),
        f"드라이버 {drv['version']} >= toolkit {tk['version']}의 최소 드라이버 {tk['min_driver']}")
    add(v(fw["torch_cuda"])[:2] <= v(drv["max_cuda"])[:2],
        f"PyTorch CUDA {fw['torch_cuda']} <= 드라이버 최대 CUDA {drv['max_cuda']}")
    add(v(tk["version"])[0] == v(fw["torch_cuda"])[0],
        f"toolkit {tk['version']}과 PyTorch CUDA {fw['torch_cuda']}의 major가 같은가(extension 빌드)")
    if tk.get("max_gcc_major"):
        add(v(cc["gcc"])[0] <= int(tk["max_gcc_major"]),
            f"gcc {cc['gcc']}가 nvcc 지원 상한 {tk['max_gcc_major']} 이하인가")
    else:
        add(None, "nvcc가 지원하는 gcc 상한이 비어 있다: CUDA 설치 가이드에서 확인한다")
    return results


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("combo.example.toml")
    combo = tomllib.loads(path.read_text(encoding="utf-8"))
    results = check(combo)
    for level, msg in results:
        print(f"[{level}] {msg}")
    counts = {k: sum(r[0] == k for r in results) for k in ("통과", "실패", "확인 필요")}
    print(f"\n통과 {counts['통과']} · 실패 {counts['실패']} · 확인 필요 {counts['확인 필요']}")
