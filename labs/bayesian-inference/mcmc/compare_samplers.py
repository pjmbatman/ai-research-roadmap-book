"""13.5 실습: 지수 감쇠 모델의 사후분포를 Metropolis–Hastings와 HMC로 샘플링하고 진단 지표를 비교한다.

모델: y_i = A exp(-k t_i) + eps_i,  eps_i ~ N(0, 0.1^2),  theta = (log A, log k) ~ N(0, I)
실행: uv run --project labs python labs/bayesian-inference/mcmc/compare_samplers.py
"""
import numpy as np

SIGMA = 0.1
rng = np.random.default_rng(0)
t_s = np.linspace(0.0, 5.0, 20)                                   # (20,) 측정 시각 [s]
y = 2.0 * np.exp(-0.8 * t_s) + SIGMA * rng.normal(size=t_s.size)  # 참값 A = 2.0, k = 0.8 [1/s]


def log_post(th):
    f = np.exp(th[0] - np.exp(th[1]) * t_s)                       # (20,) A exp(-k t)
    return -0.5 * np.sum((y - f) ** 2) / SIGMA**2 - 0.5 * th @ th


def grad_log_post(th):
    k = np.exp(th[1])
    f = np.exp(th[0] - k * t_s)
    r = (y - f) / SIGMA**2
    return np.array([r @ f, r @ (-k * t_s * f)]) - th            # d/d(log A), d/d(log k)


def metropolis(th0, step, n, rng):
    th, lp = np.array(th0, float), log_post(np.array(th0, float))
    out, acc = np.empty((n, 2)), 0
    for i in range(n):
        prop = th + step * rng.normal(size=2)
        lp1 = log_post(prop)
        if np.log(rng.random()) < lp1 - lp:
            th, lp, acc = prop, lp1, acc + 1
        out[i] = th
    return out, acc / n, n                                       # 밀도 평가 n번


def hmc(th0, eps, L, n, rng, n_adapt=0, target=0.8):
    """n_adapt번째 반복까지는 수용 확률이 target에 가깝도록 eps를 조정한다(워밍업)."""
    th, out, acc, div = np.array(th0, float), np.empty((n, 2)), 0, 0
    for i in range(n):
        r0 = rng.normal(size=2)
        th1, r = th.copy(), r0 + 0.5 * eps * grad_log_post(th)
        for j in range(L):
            th1 = th1 + eps * r
            r = r + (eps if j < L - 1 else 0.5 * eps) * grad_log_post(th1)
        log_a = (log_post(th1) - 0.5 * r @ r) - (log_post(th) - 0.5 * r0 @ r0)
        bad = not np.isfinite(log_a) or -log_a > 1000            # 에너지 오차 폭주 = divergence
        a = 0.0 if bad else min(1.0, np.exp(log_a))
        if i < n_adapt:
            eps *= np.exp(0.1 * (a - target))                    # 단순한 보폭 적응
        if bad and i >= n // 2:
            div += 1                                             # 뒤 절반(워밍업 뒤)만 센다
        if not bad and rng.random() < a:
            th, acc = th1, acc + (i >= n_adapt)
        out[i] = th
    return out, acc / (n - n_adapt), n * (L + 1), div, eps     # gradient 평가 n(L+1)번


def split_rhat(c):
    h = c.shape[1] // 2
    c = np.vstack([c[:, :h], c[:, h:2 * h]])
    n = c.shape[1]
    W, B = c.var(axis=1, ddof=1).mean(), n * c.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def ess(c):
    M, N = c.shape
    x = c - c.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, 2 * N, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :N]
    rho = (ac / ac[:, :1]).mean(axis=0)
    tau, k = -1.0, 0
    while 2 * k + 1 < N and rho[2 * k] + rho[2 * k + 1] > 0:
        tau += 2 * (rho[2 * k] + rho[2 * k + 1])
        k += 1
    return M * N / tau


def report(name, runs, n_warm):
    draws = np.array([r[0][n_warm:] for r in runs])                # (4, N, 2)
    cost = sum(r[2] for r in runs)
    acc = np.mean([r[1] for r in runs])
    A, k = np.exp(draws[..., 0]), np.exp(draws[..., 1])
    row = [f"{name:<8} 수용률 {acc:.2f}"]
    for lab, v in (("A", A), ("k", k)):
        e = ess(v)
        row.append(f"{lab}: 평균 {v.mean():.3f} sd {v.std():.3f} R-hat {split_rhat(v):.3f} ESS {e:5.0f}")
    row.append(f"ESS(k)/1000 평가 {1000 * ess(k) / cost:5.1f}")
    print("  ".join(row))


if __name__ == "__main__":
    starts = [[1.5, -1.0], [-0.5, 1.0], [1.5, 1.0], [-0.5, -1.0]]   # 흩어진 시작점
    n, warm = 4000, 2000
    rng = np.random.default_rng(1)
    report("MH", [metropolis(s, 0.1, n, rng) for s in starts], warm)
    for name, eps0, n_adapt in (("HMC-고정", 0.03, 0), ("HMC-적응", 0.03, warm)):
        runs = [hmc(s, eps0, 15, n, rng, n_adapt) for s in starts]
        report(name, runs, warm)
        print(f"{'':9}워밍업 뒤 divergence {sum(r[3] for r in runs)}회, 최종 eps {[round(float(r[4]), 3) for r in runs]}")
