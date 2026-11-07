"""23.9 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/research-practice/figures-and-tables/cells.py
그림은 임시 폴더(tempfile.gettempdir())에 저장한다.
"""
# %% cell 1 — 오차 막대는 무엇을 나타내는가
import numpy as np
from scipy import stats

rng = np.random.default_rng(2)
seed_fx = rng.normal(0, 0.03, 5)                    # seed마다 두 방법 공통 효과, (5,)
rmse_a = 0.42 + seed_fx + rng.normal(0, 0.006, 5)   # 온도 예측 RMSE (K), (5,)
rmse_b = 0.40 + seed_fx + rng.normal(0, 0.006, 5)
t4 = stats.t.ppf(0.975, df=4)                       # 자유도 4의 97.5% 분위수
for name, r in (("A", rmse_a), ("B", rmse_b)):
    sd = r.std(ddof=1)
    se = sd / np.sqrt(len(r))
    print(f"{name}: 평균 {r.mean():.3f} K   SD {sd:.3f}   SE {se:.3f}   95% CI 반폭 {t4 * se:.3f}")
d = rmse_b - rmse_a                                 # 같은 seed끼리 짝지은 차이, (5,)
half = t4 * d.std(ddof=1) / np.sqrt(len(d))
print(f"짝지은 차이 B − A: {d.mean():+.3f} K, 95% 구간 [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}]")

# %% cell 2 — 잘린 축과 lie factor
import os
import tempfile

import matplotlib
matplotlib.use("Agg")                               # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt

acc = {"A": 0.912, "B": 0.925}                      # 정확도

def lie_factor(ax, a, b):
    y0 = ax.get_ylim()[0]                           # 막대가 시작하는 축의 아래 끝
    shown = ((b - y0) - (a - y0)) / (a - y0)        # 그림에서 보이는 변화율
    return shown / ((b - a) / a)                    # ÷ 데이터의 변화율

for tag, ylim in (("truncated", (0.905, 0.93)), ("zero-based", (0.0, 1.0))):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.bar(list(acc), list(acc.values()), color="#0072B2")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Accuracy")
    fig.savefig(os.path.join(tempfile.gettempdir(), f"bars-{tag}.svg"))
    plt.close(fig)
    print(f"{tag:10s}  y축 [{ylim[0]}, {ylim[1]}]  lie factor {lie_factor(ax, *acc.values()):.1f}")

# %% cell 3 — 컬러맵의 밝기(L*)는 단조로운가
def lightness(rgb):                                 # sRGB (N, 3), 0–1 → CIELAB L* (N,)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])    # 상대 휘도, (N,)
    return np.where(Y > (6 / 29) ** 3, 116 * np.cbrt(Y) - 16, Y * (29 / 3) ** 3)

x = np.linspace(0, 1, 256)
for name in ("jet", "viridis", "gray"):
    cmap = matplotlib.colormaps[name]
    L = lightness(cmap(x)[:, :3])                    # (256,)
    up = np.diff(L) > 0.01
    print(f"{name:8s} L* {L[0]:5.1f} → {L[-1]:5.1f}  최대 L* 위치 {x[L.argmax()]:.2f}"
          f"  단조 증가 {bool(up.all())}")

okabe_ito = {"orange": "#E69F00", "sky blue": "#56B4E9", "bluish green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "reddish purple": "#CC79A7"}
rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in okabe_ito.values()])   # (7, 3)
for k, v in sorted(zip(okabe_ito, lightness(rgb)), key=lambda kv: kv[1]):
    print(f"  {k:14s} L* {v:5.1f}")

# %% cell 4 — 불확실성에 맞춘 유효숫자
def fmt(mean, err, sig=2):
    dec = max(sig - 1 - int(np.floor(np.log10(err))), 0)   # err의 유효숫자 sig개가 남는 소수 자릿수
    return f"{mean:.{dec}f} ± {err:.{dec}f}"

time_ms = rng.lognormal(np.log(12), 0.15, 5)               # 추론 시간 (ms), (5,)
rows = {"A": (rmse_a, time_ms * 0.8), "B": (rmse_b, time_ms)}
for name, (r, t) in rows.items():
    raw = f"{r.mean():.8f} ± {r.std(ddof=1):.8f}"
    print(f"{name}  RMSE 그대로 {raw}  →  {fmt(r.mean(), r.std(ddof=1))} K"
          f"   추론 시간 {fmt(t.mean(), t.std(ddof=1))} ms")
