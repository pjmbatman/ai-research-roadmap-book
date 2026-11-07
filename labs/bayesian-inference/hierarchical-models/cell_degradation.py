"""13.3 실습: 여러 배터리 셀의 열화율을 no pooling과 partial pooling으로 추정해 비교한다.

상황: 셀 12개의 용량 [mAh]을 사이클마다 잰다. 측정 계획이 셀마다 다르다.
  - 짧은 시험 셀 4개: 0–40 사이클에서 3번
  - 중간 시험 셀 4개: 0–100 사이클에서 6번
  - 긴 시험 셀 4개:  0–280 사이클에서 15번
모델: capacity_ij = c_j + r_j · cycle_ij + ε_ij,  r_j ~ N(μ, τ²),  ε_ij ~ N(0, σ²)
참값: μ = -0.40 mAh/cycle, τ = 0.05 mAh/cycle, σ = 3 mAh (알려져 있다고 둔다).

세 방법으로 셀별 열화율 r_j와 90% 구간을 구한다.
  (1) no pooling: 셀마다 최소제곱 기울기와 표준오차
  (2) partial pooling, empirical Bayes: 기울기 추정값 r̂_j ~ N(r_j, se_j²)에 대해 (μ, τ)를 주변우도 최대화로 정한다
  (3) partial pooling, 계층 베이지안: (μ, τ)에 사전분포(μ 균등, τ ~ half-Normal(0, 0.2))를 두고 격자에서 적분한다
데이터셋 500개를 새로 만들어 셀 그룹별 RMSE와 90% 구간 포함률을 잰다.
그림(데이터셋 하나의 추정값)은 outputs/cell_degradation.png에 저장한다.

실행: uv run --project labs python labs/bayesian-inference/hierarchical-models/cell_degradation.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).parent / "outputs"
MU, TAU, SIGMA = -0.40, 0.05, 3.0
PLANS = [np.linspace(0, 40, 3)] * 4 + [np.linspace(0, 100, 6)] * 4 + [np.linspace(0, 280, 15)] * 4
GROUPS = {"짧은 시험(3회)": slice(0, 4), "중간 시험(6회)": slice(4, 8), "긴 시험(15회)": slice(8, 12)}
Z90 = stats.norm.ppf(0.95)
MUS = np.linspace(-0.8, 0.0, 161)
TAUS = np.linspace(0.002, 0.4, 200)


def simulate(rng):
    r = MU + TAU * rng.normal(size=len(PLANS))
    c0 = 2500 + 10 * rng.normal(size=len(PLANS))
    return r, [c0[j] + r[j] * cyc + SIGMA * rng.normal(size=cyc.size) for j, cyc in enumerate(PLANS)]


def no_pooling(data):
    slopes, ses = [], []
    for cyc, cap in zip(PLANS, data):
        slopes.append(np.polyfit(cyc, cap, 1)[0])
        ses.append(SIGMA / np.sqrt(np.sum((cyc - cyc.mean()) ** 2)))
    return np.array(slopes), np.array(ses)


def empirical_bayes(rhat, se):
    v = TAUS[:, None] ** 2 + se**2                                       # (T, J)
    m = np.sum(rhat / v, axis=1) / np.sum(1 / v, axis=1)                 # (T,)
    ll = -0.5 * np.sum(np.log(v) + (rhat - m[:, None]) ** 2 / v, axis=1)
    k = np.argmax(ll)
    B = se**2 / (se**2 + TAUS[k] ** 2)
    mean = B * m[k] + (1 - B) * rhat
    sd = np.sqrt(1 / (1 / se**2 + 1 / TAUS[k] ** 2))
    return mean, sd, TAUS[k]


def hierarchical(rhat, se):
    M, T = np.meshgrid(MUS, TAUS, indexing="ij")                         # (A, T)
    s2 = se[:, None, None] ** 2
    loglik = stats.norm.logpdf(rhat[:, None, None], M, np.sqrt(T**2 + s2)).sum(axis=0)
    w = np.exp(loglik + stats.halfnorm.logpdf(T, scale=0.2) - (loglik + stats.halfnorm.logpdf(T, scale=0.2)).max())
    w /= w.sum()
    B = s2 / (s2 + T**2)
    cm = B * M + (1 - B) * rhat[:, None, None]                           # (J, A, T)
    cv = 1 / (1 / s2 + 1 / T**2)
    mean = np.sum(w * cm, axis=(1, 2))
    sd = np.sqrt(np.sum(w * (cv + cm**2), axis=(1, 2)) - mean**2)
    return mean, sd                                                      # 정규 근사 구간에 쓴다


def main():
    rng = np.random.default_rng(0)
    sq = {k: [] for k in ("no pooling", "partial (EB)", "partial (계층)")}
    cover = {k: [] for k in sq}
    width = {k: [] for k in sq}
    tau_hats = []
    for _ in range(500):
        r, data = simulate(rng)
        rhat, se = no_pooling(data)
        eb_m, eb_s, t_hat = empirical_bayes(rhat, se)
        hb_m, hb_s = hierarchical(rhat, se)
        tau_hats.append(t_hat)
        for name, (mean, sd) in {"no pooling": (rhat, se), "partial (EB)": (eb_m, eb_s), "partial (계층)": (hb_m, hb_s)}.items():
            sq[name].append((mean - r) ** 2)
            cover[name].append(np.abs(mean - r) <= Z90 * sd)
            width[name].append(2 * Z90 * sd)

    print(f"참값: μ = {MU} mAh/cycle, τ = {TAU}, σ = {SIGMA} mAh,  데이터셋 500개\n")
    print("셀별 열화율의 RMSE [mAh/cycle]")
    print("짧은 시험  중간 시험  긴 시험   방법")
    for name in sq:
        e = np.array(sq[name])
        print("  ".join(f"{np.sqrt(e[:, g].mean()):8.4f}" for g in GROUPS.values()) + f"   {name}")
    print("\n90% 구간의 포함률")
    print("짧은 시험  중간 시험  긴 시험   방법")
    for name in cover:
        c = np.array(cover[name])
        print("  ".join(f"{c[:, g].mean():8.3f}" for g in GROUPS.values()) + f"   {name}")
    print("\n90% 구간의 평균 폭 [mAh/cycle]")
    print("짧은 시험  중간 시험  긴 시험   방법")
    for name in width:
        w = np.array(width[name])
        print("  ".join(f"{w[:, g].mean():8.4f}" for g in GROUPS.values()) + f"   {name}")
    tau_hats = np.array(tau_hats)
    print(f"\nEB가 추정한 τ: 중앙값 {np.median(tau_hats):.3f},  0.01 미만으로 붕괴한 비율 {np.mean(tau_hats < 0.01):.2f}")

    r, data = simulate(np.random.default_rng(7))
    rhat, se = no_pooling(data)
    hb_m, hb_s = hierarchical(rhat, se)
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    j = np.arange(len(PLANS))
    ax.errorbar(j - 0.15, rhat, Z90 * se, fmt="o", label="no pooling")
    ax.errorbar(j + 0.15, hb_m, Z90 * hb_s, fmt="s", label="partial pooling")
    ax.plot(j, r, "k*", label="true")
    ax.axhline(MU, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-1.2, 0.4)
    ax.set_xlabel("cell")
    ax.set_ylabel("degradation rate [mAh/cycle]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cell_degradation.png", dpi=120)
    print(f"그림 저장: {OUT / 'cell_degradation.png'}")


if __name__ == "__main__":
    main()
