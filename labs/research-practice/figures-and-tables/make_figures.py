"""23.9 실습: 결과 파일 하나에서 논문 그림과 표를 모두 다시 만든다.

실행: uv run --project labs python labs/research-practice/figures-and-tables/make_figures.py
입력: results/runs.csv (simulate_runs.py가 만든다)
출력: outputs/fig-paired-seeds.pdf, outputs/fig-gap-by-temp.pdf, outputs/tab-per-temp.tex
같은 입력이면 같은 바이트가 나오도록 날짜 메타데이터를 지운다. 두 번 실행해 hash를 비교한다.
"""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "outputs"
COLORS = {"A": "#0072B2", "B": "#D55E00"}          # Okabe–Ito blue, vermillion
MARKERS = {"A": "o", "B": "s"}                     # 흑백에서도 구별되게 모양도 다르게
STYLE = {
    "figure.figsize": (3.25, 2.2),                 # 2단 논문의 한 단 폭(인치) — 템플릿에서 확인
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,                            # 글꼴을 TrueType으로 넣어 텍스트로 남긴다
    "savefig.bbox": "tight",
}


def load(path):
    runs = defaultdict(dict)                       # (method, temp_c) -> {seed: rmse_k}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            runs[r["method"], int(r["temp_c"])][int(r["seed"])] = float(r["rmse_k"])
    temps = sorted({t for _, t in runs})
    seeds = sorted(runs["A", temps[0]])
    arr = {m: np.array([[runs[m, t][s] for s in seeds] for t in temps]) for m in ("A", "B")}
    return temps, arr                              # arr[m]: (온도 수, seed 수)


def fmt(mean, err, sig=2):
    dec = max(sig - 1 - int(np.floor(np.log10(err))), 0)
    return f"{mean:.{dec}f} \\pm {err:.{dec}f}"


def paired_ci(d):                                  # d: (seed 수,)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half


def fig_paired_seeds(temps, arr, path):
    fig, ax = plt.subplots()
    for i, t in enumerate(temps):
        for s in range(arr["A"].shape[1]):         # 같은 seed끼리 선으로 이어 짝을 보인다
            ax.plot([i - 0.15, i + 0.15], [arr["A"][i, s], arr["B"][i, s]], color="0.75", lw=0.6, zorder=1)
        for m, dx in (("A", -0.15), ("B", 0.15)):
            ax.scatter(np.full(arr[m].shape[1], i + dx), arr[m][i], s=12, color=COLORS[m],
                       marker=MARKERS[m], label=m if i == 0 else None, zorder=2)
    ax.set_xticks(range(len(temps)), [f"{t} °C" for t in temps])
    ax.set_xlabel("Test temperature")
    ax.set_ylabel("RMSE (K)")
    ax.legend(frameon=False)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def fig_gap_by_temp(temps, arr, path):
    fig, ax = plt.subplots()
    for i, t in enumerate(temps):
        mean, lo, hi = paired_ci(arr["B"][i] - arr["A"][i])
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="k", capsize=3)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(range(len(temps)), [f"{t} °C" for t in temps])
    ax.set_xlabel("Test temperature")
    ax.set_ylabel("B − A RMSE (K)\n(mean, 95% t-CI, 5 seeds)")
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def table_per_temp(temps, arr, path):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Temp. & A (K) & B (K) & B $-$ A (K), 95\% CI \\", r"\midrule"]
    for i, t in enumerate(temps):
        a, b = arr["A"][i], arr["B"][i]
        mean, lo, hi = paired_ci(b - a)
        gap = f"{mean:+.3f}\\ [{lo:+.3f}, {hi:+.3f}]"   # 수식 모드: 음수 부호가 하이픈이 아니라 minus로
        if hi < 0 or lo > 0:                       # 구간이 0을 포함하지 않을 때만 굵게
            gap = r"\mathbf{" + gap + "}"
        lines.append(f"{t}\\,°C & ${fmt(a.mean(), a.std(ddof=1))}$ & ${fmt(b.mean(), b.std(ddof=1))}$ & ${gap}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build():
    temps, arr = load(HERE / "results" / "runs.csv")
    OUT.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        fig_paired_seeds(temps, arr, OUT / "fig-paired-seeds.pdf")
        fig_gap_by_temp(temps, arr, OUT / "fig-gap-by-temp.pdf")
    table_per_temp(temps, arr, OUT / "tab-per-temp.tex")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted(OUT.iterdir())}


if __name__ == "__main__":
    first, second = build(), build()
    for name, h in first.items():
        print(f"{name:22s} sha256:{h}  다시 만들어도 같음: {second[name] == h}")
    print((OUT / "tab-per-temp.tex").read_text(encoding="utf-8"), end="")
