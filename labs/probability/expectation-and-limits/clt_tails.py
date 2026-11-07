"""11.5 실습: 표본 평균의 근사 95% 구간이 분포와 표본 크기에 따라 얼마나 맞는지 확인한다.

각 분포에서 크기 n인 표본을 reps번 뽑아, 구간 x̄ ± 1.96·s/√n이 참 평균을 포함하는 비율(포함률)을 센다.
중심극한정리가 성립하고 n이 충분히 크면 포함률은 0.95에 가까워야 한다.

실행: uv run --project labs python labs/probability/expectation-and-limits/clt_tails.py
"""
import numpy as np

SEED, REPS = 0, 10_000
NS = (5, 30, 300, 3000)

# 이름, 표본 생성 함수, 참 평균(없으면 None)
DISTS = [
    ("정규 N(0,1)", lambda r, s: r.normal(size=s), 0.0),
    ("지수 Exp(1)", lambda r, s: r.exponential(size=s), 1.0),
    ("로그정규 σ=1.5", lambda r, s: r.lognormal(0.0, 1.5, size=s), np.exp(1.5**2 / 2)),
    ("Pareto α=2.5", lambda r, s: r.pareto(2.5, size=s) + 1.0, 2.5 / 1.5),     # 분산 유한
    ("Pareto α=1.5", lambda r, s: r.pareto(1.5, size=s) + 1.0, 1.5 / 0.5),     # 분산 무한
    ("Cauchy", lambda r, s: r.standard_cauchy(size=s), None),                 # 평균 없음
]


def coverage(sample, mu, n, rng):
    x = sample(rng, (REPS, n))                       # (REPS, n)
    m = x.mean(axis=1)                               # (REPS,)
    se = x.std(axis=1, ddof=1) / np.sqrt(n)          # (REPS,)
    return np.mean(np.abs(m - mu) <= 1.96 * se), m


def main():
    rng = np.random.default_rng(SEED)
    print(f"근사 95% 구간 x̄ ± 1.96·s/√n 의 포함률 (시드 {SEED}, 반복 {REPS}번)")
    print("".join(f"n={n:<5d}" for n in NS) + "분포")
    spreads = {}
    for name, sample, mu in DISTS:
        row = []
        for n in NS:
            if mu is None:
                _, m = coverage(sample, 0.0, n, rng)
                spreads[n] = np.subtract(*np.quantile(m, [0.75, 0.25]))
                row.append("  -    ")
            else:
                cov, _ = coverage(sample, mu, n, rng)
                row.append(f"{cov:.3f}  ")
        print("".join(row) + name)
    print()
    print("Cauchy 표본 평균의 사분위 범위(IQR): " + ", ".join(f"n={n} {v:.2f}" for n, v in spreads.items()))


if __name__ == "__main__":
    main()
