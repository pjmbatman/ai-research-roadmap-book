"""방사성 붕괴 계수 자료에서 반감기를 추정한다 (4.4 실습용 최소 연구 프로젝트).

모형: counts(t) = N0 * exp(-lambda * t) + 잡음,  반감기 t_half = ln 2 / lambda
실행: python src/halflife.py --smoke
"""
import argparse

import numpy as np


def simulate(half_life_s, n0=1e4, t_max_s=600.0, n=61, seed=0):
    """Poisson 잡음이 있는 계수 자료를 만든다. 반환: t_s (n,), counts (n,)"""
    rng = np.random.default_rng(seed)
    t_s = np.linspace(0.0, t_max_s, n)                          # (n,)
    rate = n0 * np.exp(-np.log(2) / half_life_s * t_s)          # (n,)
    return t_s, rng.poisson(rate).astype(float)


def fit_half_life(t_s, counts):
    """log(counts)를 t에 대해 가중 최소제곱으로 맞춰 반감기(초)를 반환한다. counts > 0인 점만 쓴다."""
    m = counts > 0
    t, c = t_s[m], counts[m]
    W = np.sqrt(c)                                              # Var[log c] ≈ 1/c → 가중치 √c
    A = np.column_stack([np.ones_like(t), t]) * W[:, None]      # (n, 2)
    coef, *_ = np.linalg.lstsq(A, np.log(c) * W, rcond=None)
    return np.log(2) / -coef[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="작은 합성 자료로 1초 안에 끝나는 점검")
    p.add_argument("--half-life-s", type=float, default=120.0)
    args = p.parse_args()
    t_s, counts = simulate(args.half_life_s)
    est = fit_half_life(t_s, counts)
    print(f"참값 {args.half_life_s:.1f} s, 추정 {est:.1f} s")
    if args.smoke:
        assert abs(est - args.half_life_s) / args.half_life_s < 0.05, "smoke 실패: 상대 오차 5% 초과"
        print("smoke OK")


if __name__ == "__main__":
    main()
