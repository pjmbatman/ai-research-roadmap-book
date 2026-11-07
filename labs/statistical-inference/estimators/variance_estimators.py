"""12.1 실습: 분산의 편향·불편 추정량을 시뮬레이션해 bias·variance·MSE를 비교한다.

편차 제곱합 SS = Σ(x_i - x̄)²를 n-1, n, n+1로 나눈 세 추정량을 표본 크기와 분포를 바꿔 가며 비교하고,
MSE를 가장 작게 만드는 나누는 수 d* = E[SS²] / (σ² E[SS])도 시뮬레이션으로 구한다.

실행: uv run --project labs python labs/statistical-inference/estimators/variance_estimators.py
"""
import numpy as np

R = 200_000  # 반복 실험 횟수


def sampler(name, rng, size):
    """분산이 1이 되도록 맞춘 세 분포."""
    if name == "정규":
        return rng.normal(0.0, 1.0, size=size)
    if name == "균등":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=size)   # 분산 1
    if name == "Laplace":
        return rng.laplace(0.0, 1 / np.sqrt(2), size=size)        # 분산 2b² = 1
    raise ValueError(name)


def main():
    rng = np.random.default_rng(0)
    sigma2 = 1.0
    print("분포      n   나누는 수   bias      variance   MSE      | MSE 최소 나누는 수 d*")
    for name in ("균등", "정규", "Laplace"):
        for n in (5, 20):
            X = sampler(name, rng, (R, n))                                  # (R, n)
            ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)     # (R,)
            d_star = np.mean(ss**2) / (sigma2 * np.mean(ss))
            for i, d in enumerate((n - 1, n, n + 1)):
                est = ss / d
                bias, var = est.mean() - sigma2, est.var()
                mse = np.mean((est - sigma2) ** 2)
                tail = f"| {d_star:.2f}" if i == 0 else "|"
                print(f"{name:8s} {n:3d}   {d:3d}        {bias:+.4f}   {var:.4f}     {mse:.4f}   {tail}")
        print()


if __name__ == "__main__":
    main()
