"""5.3 실습: 같은 계산을 Python loop와 벡터화로 작성해 시간·메모리·수치 결과를 비교한다.

실행: uv run --project labs python labs/python-and-computing/scientific-python/loop_vs_vectorized.py

계산: N개 입자의 Lennard-Jones 퍼텐셜 에너지(무차원 단위, epsilon = sigma = 1)
    E = sum_{i<j} 4 (r_ij^-12 - r_ij^-6)
세 구현을 비교한다.
    loop      : 이중 Python 반복문
    broadcast : (N, N, D) 차이 배열을 만드는 broadcasting
    expand    : |x|^2 - 2 x.y + |y|^2 전개(행렬 곱 하나, (N, N) 메모리)
시간은 3회 측정의 최솟값, 메모리는 tracemalloc으로 잰 최대 추가 할당량이다. 시간 값은 기계마다 다르다.
"""
import time
import tracemalloc

import numpy as np


def lj_loop(P):
    n = len(P)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = float(np.sum((P[i] - P[j]) ** 2))
            inv6 = 1.0 / r2**3
            E += 4.0 * (inv6 * inv6 - inv6)
    return E


def lj_broadcast(P):
    d = P[:, None, :] - P[None, :, :]          # (N, N, D)
    r2 = np.einsum("ijk,ijk->ij", d, d)        # (N, N)
    iu = np.triu_indices(len(P), k=1)
    inv6 = 1.0 / r2[iu] ** 3                   # (N(N-1)/2,)
    return float(np.sum(4.0 * (inv6 * inv6 - inv6)))


def lj_expand(P):
    sq = np.sum(P * P, axis=1)                 # (N,)
    r2 = sq[:, None] - 2.0 * (P @ P.T) + sq[None, :]   # (N, N)
    iu = np.triu_indices(len(P), k=1)
    inv6 = 1.0 / r2[iu] ** 3
    return float(np.sum(4.0 * (inv6 * inv6 - inv6)))


def lattice(n_side, spacing, rng, dtype):
    """단순 입방 격자에 작은 흔들림을 준 입자 배치. (n_side^3, 3)"""
    g = np.arange(n_side) * spacing
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return (P + 0.05 * rng.normal(size=P.shape)).astype(dtype)


def measure(fn, P, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        E = fn(P)
        best = min(best, time.perf_counter() - t0)
    tracemalloc.start()
    fn(P)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return E, best, peak


def main():
    rng = np.random.default_rng(0)
    P = lattice(8, 1.12, rng, np.float64)      # N = 512
    print(f"N = {len(P)} 입자, float64")
    print(f"{'구현':<10}{'에너지':>16}{'시간(s)':>10}{'추가 메모리(MB)':>18}")
    results = {}
    for name, fn in [("loop", lj_loop), ("broadcast", lj_broadcast), ("expand", lj_expand)]:
        E, t, peak = measure(fn, P, repeat=1 if name == "loop" else 3)
        results[name] = (E, t)
        print(f"{name:<10}{E:>16.6f}{t:>10.4f}{peak / 1e6:>18.2f}")
    E_ref = results["loop"][0]
    for name in ("broadcast", "expand"):
        rel = abs(results[name][0] - E_ref) / abs(E_ref)
        print(f"{name} 상대 오차(loop 기준): {rel:.1e},  속도 향상: {results['loop'][1] / results[name][1]:.0f}배")

    P32 = P.astype(np.float32)
    for name, fn in [("broadcast", lj_broadcast), ("expand", lj_expand)]:
        rel = abs(fn(P32) - E_ref) / abs(E_ref)
        print(f"float32 {name} 상대 오차: {rel:.1e}")


if __name__ == "__main__":
    main()
