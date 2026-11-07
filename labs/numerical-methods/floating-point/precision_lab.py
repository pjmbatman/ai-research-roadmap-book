"""9.1 실습: softmax 계산식과 합의 순서가 수치 오차에 주는 영향을 비교한다.

1) logits의 크기를 키우며 naive softmax와 log-sum-exp softmax를 float32·float16으로 계산해
   float64 기준값과의 오차, nan 여부를 본다.
2) 크기가 여러 자릿수에 걸친 float32 배열을 여러 순서·방법으로 더해 상대 오차를 비교한다.

실행: uv run --project labs python labs/numerical-methods/floating-point/precision_lab.py
"""
import math

import numpy as np


def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()


def softmax_stable(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def softmax_error(fn, z, dtype):
    ref = softmax_stable(z.astype(np.float64))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        out = fn(z.astype(dtype)).astype(np.float64)
    if not np.all(np.isfinite(out)):
        return "nan"
    return f"{np.max(np.abs(out - ref)):.1e}"


def kahan(a):
    s, c = a.dtype.type(0), a.dtype.type(0)
    for v in a:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def sequential(a):
    s = a.dtype.type(0)
    for v in a:
        s += v
    return s


def main():
    rng = np.random.default_rng(0)
    base = rng.normal(size=10)                                 # (10,) logits의 모양
    print("[1] softmax: float64 기준값과의 최대 절대 오차 (nan = 계산 실패)")
    print(f"{'logit 크기':>10} {'naive f32':>10} {'stable f32':>11} {'naive f16':>10} {'stable f16':>11}")
    for scale in [1, 3, 10, 30, 100]:
        z = base * scale
        cols = [softmax_error(f, z, d) for d in (np.float32, np.float16)
                for f in (softmax_naive, softmax_stable)]
        print(f"{scale:>10} {cols[0]:>10} {cols[1]:>11} {cols[2]:>10} {cols[3]:>11}")

    print()
    print("[2] float32 합 20만 개: 크기가 1e-3 ~ 1e3에 걸친 양수 (기준: math.fsum)")
    a = (10.0 ** rng.uniform(-3, 3, 200_000)).astype(np.float32)   # (200000,)
    exact = math.fsum(a.astype(np.float64))
    methods = {
        "원래 순서": sequential(a),
        "오름차순(작은 것부터)": sequential(np.sort(a)),
        "내림차순(큰 것부터)": sequential(np.sort(a)[::-1]),
        "np.sum (쌍별 합)": np.sum(a),
        "Kahan (원래 순서)": kahan(a),
        "float64로 누적": sequential(a.astype(np.float64)),
    }
    for name, s in methods.items():
        print(f"  {name:<20} 상대 오차 = {abs(float(s) - exact) / exact:.1e}")


if __name__ == "__main__":
    main()
