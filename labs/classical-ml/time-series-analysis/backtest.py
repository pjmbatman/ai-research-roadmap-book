"""16.7 실습: 시계열 예측을 rolling-origin backtesting으로 평가하고 horizon별 오차를 분석한다.

합성 데이터: 시간당 기온 120일. 선형 추세, 하루 주기, AR(1) 잡음에 더해 80일째에 수준이 +2 °C
바뀐다(센서 이전·기후 체제 변화 같은 구조 변화). 원점을 하루씩 옮기며 48시간 앞까지 예측하고,
horizon별 RMSE와 원점 사이의 변동, 구조 변화 전후의 오차를 비교한다.

실행: uv run --project labs python labs/classical-ml/time-series-analysis/backtest.py
"""
import numpy as np

SEED, DAYS, SHIFT_DAY, H = 0, 120, 80, 48


def make_series(rng):
    n = 24 * DAYS
    t = np.arange(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.5)
    level = np.where(t >= 24 * SHIFT_DAY, 2.0, 0.0)                     # 구조 변화
    return 15 + 0.05 * t / 24 + 3 * np.sin(2 * np.pi * t / 24) + level + ar


def fit_ar1(e):
    return float(e[1:] @ e[:-1] / (e[:-1] @ e[:-1]))


def season_ar_forecast(y, o, h, window=None):
    """시점 o까지(창이 있으면 마지막 window시간만) 보고 선형 추세 + 시각별 평균 + AR(1)로 h시간 예측."""
    start = 0 if window is None else max(0, o + 1 - window)
    th = np.arange(start, o + 1)
    c = np.polyfit(th, y[start:o + 1], 1)
    d = y[start:o + 1] - np.polyval(c, th)
    s = np.array([d[th % 24 == k].mean() for k in range(24)])
    e = d - s[th % 24]
    fut = o + np.arange(1, h + 1)
    return np.polyval(c, fut) + s[fut % 24] + fit_ar1(e) ** np.arange(1, h + 1) * e[-1]


def main():
    y = make_series(np.random.default_rng(SEED))
    origins = np.arange(24 * 60 - 1, len(y) - H, 24)                   # 60일째부터 하루 간격
    models = {
        "persistence": lambda o: np.full(H, y[o]),
        "seasonal naive (전날 같은 시각)": lambda o: y[o + 1 + np.arange(H) % 24 - 24],
        "추세+계절+AR(1), 확장 창": lambda o: season_ar_forecast(y, o, H),
        "추세+계절+AR(1), 최근 14일 창": lambda o: season_ar_forecast(y, o, H, window=24 * 14),
    }
    errs = {name: np.array([f(o) - y[o + 1:o + 1 + H] for o in origins]) for name, f in models.items()}  # (원점, H)

    print(f"원점 {len(origins)}개 (60일째부터 하루 간격), horizon 1–{H}시간, 구조 변화 {SHIFT_DAY}일째 +2 °C")
    hs = (1, 6, 12, 24, 48)
    print("\n1. horizon별 RMSE [°C]  (괄호: 원점별 제곱오차 평균의 표준오차로 근사한 RMSE의 ±)")
    print(" " * 32 + "".join(f"h={h:<9d}" for h in hs))
    for name, e in errs.items():
        se2 = e ** 2                                                    # (원점, H)
        cells = []
        for h in hs:
            m = se2[:, h - 1].mean()
            se = se2[:, h - 1].std(ddof=1) / np.sqrt(len(origins))
            cells.append(f"{np.sqrt(m):.2f}(±{se / (2 * np.sqrt(m)):.2f})")
        print(f"{name:30s}  " + " ".join(f"{c:11s}" for c in cells))

    print("\n2. 구조 변화 전후의 RMSE (h = 1–24 평균) [°C]")
    after = origins >= 24 * SHIFT_DAY
    for name, e in errs.items():
        r = lambda m: np.sqrt(np.mean(e[m, :24] ** 2))
        print(f"{name:30s}  변화 전 {r(~after):.2f}   변화 후 {r(after):.2f}")

    print("\n3. 이웃 원점의 오차 상관 (확장 창 모델, h = 24)")
    e = errs["추세+계절+AR(1), 확장 창"][:, 23]
    print(f"   하루 간격 원점 사이 상관 {np.corrcoef(e[:-1], e[1:])[0, 1]:.2f}")
    e1 = errs["추세+계절+AR(1), 확장 창"][:, 0]
    print(f"   (h = 1에서는 {np.corrcoef(e1[:-1], e1[1:])[0, 1]:.2f})")


if __name__ == "__main__":
    main()
