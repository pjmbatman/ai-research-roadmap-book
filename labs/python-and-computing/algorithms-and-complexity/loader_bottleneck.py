"""5.4 실습: 데이터 로더의 병목을 profiling으로 찾아 개선하고, 개선 전후를 숫자로 보고한다.

실행: uv run --project labs python labs/python-and-computing/algorithms-and-complexity/loader_bottleneck.py

가상의 로더는 임시 디렉터리의 샘플 파일(.npy)과 라벨 파일(JSON)을 읽어 batch를 만든다. 느린 버전에는 두 병목이 숨어 있다.
  1. 샘플마다 라벨 파일 전체를 다시 읽고 파싱한다 (반복 파일 읽기)
  2. 이미 본 샘플 ID를 리스트에 기록하고 membership을 검사한다 (O(n^2) 중복 탐색)
cProfile로 tottime 상위 함수를 출력해 병목을 찾고, 고친 버전과 시간을 비교한다. 시간 값은 기계마다 다르다.
"""
import cProfile
import json
import pstats
import tempfile
import time
from pathlib import Path

import numpy as np

N_SAMPLES = 3_000


def make_dataset(root: Path, rng) -> None:
    for i in range(N_SAMPLES):
        np.save(root / f"s{i:05d}.npy", rng.normal(size=16).astype(np.float32))
    labels = {f"s{i:05d}": int(i % 5) for i in range(N_SAMPLES)}
    (root / "labels.json").write_text(json.dumps(labels))


def load_labels(root: Path) -> dict:
    return json.loads((root / "labels.json").read_text())


def slow_loader(root: Path, ids: list[str]):
    seen = []
    xs, ys = [], []
    for sid in ids:
        if sid in seen:                          # 병목 2: 리스트 membership
            continue
        seen.append(sid)
        labels = load_labels(root)               # 병목 1: 샘플마다 라벨 파일 전체를 다시 읽음
        xs.append(np.load(root / f"{sid}.npy"))
        ys.append(labels[sid])
    return np.stack(xs), np.array(ys)


def fast_loader(root: Path, ids: list[str]):
    labels = load_labels(root)                   # 한 번만 읽는다
    seen = set()
    xs, ys = [], []
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        xs.append(np.load(root / f"{sid}.npy"))
        ys.append(labels[sid])
    return np.stack(xs), np.array(ys)


def timed(fn, *args, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, rng)
        ids = [f"s{i:05d}" for i in rng.integers(0, N_SAMPLES, size=2 * N_SAMPLES)]   # 중복 포함

        print("== cProfile: slow_loader, tottime 상위 5개 ==")
        prof = cProfile.Profile()
        prof.runcall(slow_loader, root, ids)
        pstats.Stats(prof).strip_dirs().sort_stats("tottime").print_stats(5)

        (x_slow, y_slow), t_slow = timed(slow_loader, root, ids, repeat=1)
        (x_fast, y_fast), t_fast = timed(fast_loader, root, ids)
        print(f"결과 동일: {np.array_equal(x_slow, x_fast) and np.array_equal(y_slow, y_fast)}  "
              f"샘플 {len(y_fast)}개")
        print(f"slow {t_slow:.3f} s  →  fast {t_fast:.3f} s  ({t_slow / t_fast:.0f}배)")


if __name__ == "__main__":
    main()
