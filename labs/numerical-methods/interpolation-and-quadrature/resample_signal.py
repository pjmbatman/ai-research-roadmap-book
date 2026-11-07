"""9.5 실습: 불규칙하게 측정된 신호를 여러 보간법으로 재표본화하고, 측정 구간 안(보간)과 밖(외삽)의 오차를 비교한다.

실행: uv run --project labs python labs/numerical-methods/interpolation-and-quadrature/resample_signal.py
"""
import numpy as np
from scipy.interpolate import Akima1DInterpolator, BarycentricInterpolator, CubicSpline, PchipInterpolator


def true_signal(t_s):
    """감쇠 진동 신호(참값). 실제 측정에서는 알 수 없다."""
    return np.exp(-0.2 * t_s) * np.sin(2.0 * t_s) + 0.3


def measure(rng, n=40, t_end_s=10.0, noise=0.01):
    """[0, t_end_s] 구간의 불규칙한 시각 n개에서 잡음 섞인 표본을 얻는다. 양 끝점은 항상 측정한다."""
    t_s = np.sort(np.concatenate([[0.0, t_end_s], rng.uniform(0.0, t_end_s, n - 2)]))  # (n,)
    return t_s, true_signal(t_s) + noise * rng.normal(size=n)


def methods(t_s, y):
    return {
        "선형(np.interp)": lambda q: np.interp(q, t_s, y),   # 구간 밖에서는 끝값을 그대로 쓴다
        "cubic spline": CubicSpline(t_s, y),                 # 구간 밖에서는 끝 조각의 3차식을 연장
        "PCHIP": PchipInterpolator(t_s, y),
        "Akima": Akima1DInterpolator(t_s, y, extrapolate=True),
        "고차 다항식": BarycentricInterpolator(t_s, y),        # n-1차 다항식 하나
    }


def main():
    t_in = np.arange(0.0, 10.0 + 1e-9, 0.05)    # (201,) 재표본화 격자, 측정 구간 안
    t_out = np.array([10.25, 10.5, 11.0])       # (3,) 측정 구간 밖

    rng = np.random.default_rng(0)
    t_s, y = measure(rng)
    print(f"측정점 {len(t_s)}개, 간격 최소 {np.min(np.diff(t_s)):.3f} s, 최대 {np.max(np.diff(t_s)):.3f} s")
    print(f"{'방법':<16}{'안 RMS':>10}{'안 최대':>10}" + "".join(f"{f't={t:.2f}s':>11}" for t in t_out))
    for name, interp in methods(t_s, y).items():
        e_in = interp(t_in) - true_signal(t_in)
        e_out = np.abs(interp(t_out) - true_signal(t_out))
        print(f"{name:<16}{np.sqrt(np.mean(e_in**2)):10.1e}{np.max(np.abs(e_in)):10.1e}"
              + "".join(f"{e:11.1e}" for e in e_out))

    # 측정 시각과 잡음을 바꾸어 20번 반복한다. 한 번의 결과에 기대지 않는다.
    stats = {}
    for seed in range(20):
        t_s, y = measure(np.random.default_rng(seed))
        for name, interp in methods(t_s, y).items():
            rms_in = np.sqrt(np.mean((interp(t_in) - true_signal(t_in)) ** 2))
            err_out = abs(interp(10.5) - true_signal(10.5))
            stats.setdefault(name, []).append((rms_in, err_out))
    print("\n20회 반복의 중앙값")
    for name, v in stats.items():
        v = np.array(v)   # (20, 2)
        print(f"{name:<16} 안 RMS {np.median(v[:, 0]):8.1e}   t=10.50s 오차 {np.median(v[:, 1]):8.1e}")


if __name__ == "__main__":
    main()
