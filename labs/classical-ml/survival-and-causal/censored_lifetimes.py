"""16.8 실습: 실험이 중단된 셀을 포함한 수명 데이터를 단순 회귀와 생존분석으로 각각 모델링해 비교한다.

합성 데이터: 셀 120개의 사이클 수명이 온도와 C-rate에 따라 달라지는 Weibull 분포(AFT 형태)를 따른다.
시험 장비 사정으로 셀마다 200–1200 cycle 사이의 임의 시점에 시험이 끝나고, 그때까지 고장 나지 않은
셀은 중도절단된다. 같은 실험을 200번 반복해 네 방법의 온도 효과 추정값 분포를 비교한다.

  (1) 고장 난 셀만으로 log 수명을 선형 회귀
  (2) 중도절단 시점을 고장으로 취급해 선형 회귀
  (3) 중도절단을 likelihood에 넣은 Weibull AFT 최대우도
  (4) Cox 비례위험 모델(부분 likelihood). Weibull AFT에서 온도 계수 b는 위험비 계수 -k·b에 대응

실행: uv run --project labs python labs/classical-ml/survival-and-causal/censored_lifetimes.py
"""
import numpy as np
from scipy.optimize import minimize

N_CELLS, N_REPS, SHAPE = 120, 200, 3.0
B_TEMP, B_CRATE = -0.03, -0.25                       # log 수명에 대한 효과: 1 °C당, 1C당


def simulate(rng):
    temp = rng.uniform(15, 45, N_CELLS) - 30         # 중심화한 온도 [°C]
    crate = rng.uniform(0.5, 2.0, N_CELLS) - 1.25    # 중심화한 C-rate [1/h]
    X = np.c_[temp, crate]                           # (N, 2)
    life = np.exp(np.log(900) + X @ [B_TEMP, B_CRATE]) * rng.weibull(SHAPE, N_CELLS)
    stop = rng.uniform(200, 1200, N_CELLS)
    return X, np.minimum(life, stop), life <= stop


def ols_temp(X, logt):
    A = np.c_[np.ones(len(logt)), X]
    return np.linalg.lstsq(A, logt, rcond=None)[0][1]


def weibull_aft(X, t, e):
    def nll(th):
        k = np.exp(th[-1])
        lam = np.exp(th[0] + X @ th[1:3])
        z = (t / lam) ** k
        return -np.sum(np.where(e, np.log(k) - np.log(lam) + (k - 1) * np.log(t / lam) - z, -z))
    x0 = np.r_[np.log(t.mean()), 0.0, 0.0, 0.0]
    return minimize(nll, x0, method="BFGS").x        # (a, b_temp, b_crate, log k)


def cox_temp(X, t, e):
    order = np.argsort(t)
    X, t, e = X[order], t[order], e[order]

    def nll(beta):                                   # Breslow 부분 likelihood (동점 없음)
        eta = X @ beta
        log_risk = np.log(np.cumsum(np.exp(eta)[::-1])[::-1])   # 위험 집합 {j : t_j >= t_i}
        return -np.sum((eta - log_risk)[e])
    return minimize(nll, np.zeros(2), method="BFGS").x[0]


def main():
    rng = np.random.default_rng(0)
    est = {k: [] for k in ("고장만 회귀", "중도절단=고장 회귀", "Weibull AFT MLE", "Cox (b로 환산)")}
    cens = []
    for _ in range(N_REPS):
        X, t, e = simulate(rng)
        cens.append(np.mean(~e))
        est["고장만 회귀"].append(ols_temp(X[e], np.log(t[e])))
        est["중도절단=고장 회귀"].append(ols_temp(X, np.log(t)))
        th = weibull_aft(X, t, e)
        b, k = th[1], np.exp(th[3])
        est["Weibull AFT MLE"].append(b)
        est["Cox (b로 환산)"].append(-cox_temp(X, t, e) / k)   # 위험비 계수 → AFT 계수 (k는 AFT 추정값)

    print(f"셀 {N_CELLS}개 × 반복 {N_REPS}번, 중도절단 비율 평균 {np.mean(cens):.0%}")
    print(f"온도 계수 b (참값 {B_TEMP:+.3f}, 10 °C 상승 시 수명 배율 {np.exp(10 * B_TEMP):.3f})")
    print("  방법                    b 평균    b 표준편차   평균 배율   |b - 참값| ≤ 2sd인 반복 비율")
    for name, v in est.items():
        v = np.array(v)
        cover = np.mean(np.abs(v - B_TEMP) <= 2 * v.std(ddof=1))
        print(f"  {name:20s}  {v.mean():+.4f}    {v.std(ddof=1):.4f}      {np.exp(10 * v.mean()):.3f}      {cover:.2f}")

    print("\n하나의 실험(첫 반복과 같은 시드)에서 45 °C, 1C 셀의 중앙 수명 예측 [cycle]")
    rng = np.random.default_rng(0)
    X, t, e = simulate(rng)
    x = np.array([15.0, -0.25])                       # 45 °C, 1C (중심화)
    true_med = np.exp(np.log(900) + x @ [B_TEMP, B_CRATE]) * np.log(2) ** (1 / SHAPE)
    for name, keep, logt in [("고장만 회귀", e, np.log(t[e])), ("중도절단=고장 회귀", np.ones_like(e), np.log(t))]:
        A = np.c_[np.ones(keep.sum()), X[keep]]
        coef = np.linalg.lstsq(A, logt, rcond=None)[0]
        print(f"  {name:20s} {np.exp(coef[0] + x @ coef[1:]):6.0f}   (log 수명 평균의 역변환)")
    th = weibull_aft(X, t, e)
    med = np.exp(th[0] + x @ th[1:3]) * np.log(2) ** (1 / np.exp(th[3]))
    print(f"  {'Weibull AFT MLE':20s} {med:6.0f}")
    print(f"  {'참값':20s} {true_med:6.0f}")


if __name__ == "__main__":
    main()
