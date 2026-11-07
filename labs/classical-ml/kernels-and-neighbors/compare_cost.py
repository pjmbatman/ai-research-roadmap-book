"""16.3 실습: 같은 비선형 데이터에 kNN·커널 릿지·GP를 적합해 예측 정확도와 계산 비용을 비교한다.

실행: uv run --project labs python labs/classical-ml/kernels-and-neighbors/compare_cost.py
시간은 기계와 BLAS 스레드 수에 따라 달라진다. 비교할 것은 N이 두 배가 될 때 시간이 몇 배가 되는가다.
스레드 경합을 줄이려면 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1을 앞에 붙여 실행한다.
"""
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor


def make_data(n, rng):
    X = rng.uniform(-3, 3, (n, 2))                                    # (n, 2)
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + rng.normal(0, 0.1, n)
    return X, y


def timed(model, Xa, ya, Xb):
    t0 = time.perf_counter()
    model.fit(Xa, ya)
    pred = model.predict(Xb)
    return pred, time.perf_counter() - t0


def main():
    rng = np.random.default_rng(0)
    Xb, _ = make_data(2000, rng)                                      # 시험 입력, 참 함수와 비교
    f_true = np.sin(Xb[:, 0]) * np.cos(Xb[:, 1])
    models = {
        "kNN (k=10)": lambda: KNeighborsRegressor(10),
        "커널 릿지 (고정 커널)": lambda: KernelRidge(kernel="rbf", gamma=0.5, alpha=0.01),
        "GP (하이퍼파라미터 최적화)": lambda: GaussianProcessRegressor(RBF(1.0) + WhiteKernel(0.01), random_state=0),
    }
    print(f"{'N':>5s}  " + "  ".join(f"{k}: RMSE / 시간[s]" for k in models))
    for n in (250, 500, 1000, 2000):
        Xa, ya = make_data(n, np.random.default_rng(n))
        row = []
        for name, make in models.items():
            if name.startswith("GP") and n > 1000:
                row.append("(생략)")
                continue
            pred, sec = timed(make(), Xa, ya, Xb)
            row.append(f"{np.sqrt(np.mean((pred - f_true) ** 2)):.3f} / {sec:6.2f}")
        print(f"{n:5d}  " + "   ".join(row))


if __name__ == "__main__":
    main()
