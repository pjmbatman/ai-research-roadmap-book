"""13.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/bayesian-inference/mcmc/cells.py
"""
# %% cell 1 — Metropolis–Hastings와 보폭
import numpy as np

rho = 0.95
P = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))   # (2, 2) 목표의 정밀도 행렬
def log_p(th):                                          # 정규화 상수를 모르는 로그 밀도
    return -0.5 * th @ P @ th

def metropolis(log_p, th0, step, n, rng):
    th = np.array(th0, dtype=float)
    lp, out, acc = log_p(th), np.empty((n, th.size)), 0  # out: (n, 2)
    for t in range(n):
        prop = th + step * rng.normal(size=th.size)      # 대칭 제안 → q가 약분된다
        lp_prop = log_p(prop)
        if np.log(rng.random()) < lp_prop - lp:          # 수용 확률 min(1, p'/p)
            th, lp, acc = prop, lp_prop, acc + 1
        out[t] = th
    return out, acc / n

rng = np.random.default_rng(0)
for step in (0.05, 0.5, 5.0):
    s, a = metropolis(log_p, [3.0, -3.0], step, 5000, rng)
    m, sd = s[1000:].mean(0), s[1000:].std(0)            # 앞 1000개는 버린다
    print(f"보폭 {step:4}: 수용률 {a:.2f}  평균 {m.round(2)}  표준편차 {sd.round(2)}")

# %% cell 2 — Gibbs sampling과 상관
def gibbs(n, rho, rng):
    out, x1, x2 = np.empty((n, 2)), 3.0, -3.0
    sd = np.sqrt(1 - rho**2)
    for t in range(n):
        x1 = rho * x2 + sd * rng.normal()               # x1 ~ p(x1 | x2)
        x2 = rho * x1 + sd * rng.normal()               # x2 ~ p(x2 | x1)
        out[t] = x1, x2
    return out

def lag1(x):                                            # 1-시차 자기상관
    x = x - x.mean()
    return (x[:-1] @ x[1:]) / (x @ x)

for r in (0.5, 0.95, 0.99):
    s = gibbs(20000, r, np.random.default_rng(1))[1000:]
    print(f"rho {r}: x1의 1-시차 자기상관 {lag1(s[:, 0]):.3f}  (rho^2 = {r**2:.3f})")

# %% cell 3 — HMC: leapfrog와 에너지 오차
grad = lambda th: -P @ th                               # d log p / d theta

def leapfrog(th, r, eps, L):
    r = r + 0.5 * eps * grad(th)                        # 운동량 반 걸음
    for i in range(L):
        th = th + eps * r                               # 위치 한 걸음
        r = r + (eps if i < L - 1 else 0.5 * eps) * grad(th)
    return th, r

def hmc(th0, eps, L, n, rng):
    th, out, acc, dH = np.array(th0, dtype=float), np.empty((n, 2)), 0, []
    for t in range(n):
        r0 = rng.normal(size=2)                         # 운동량 재추출
        th1, r1 = leapfrog(th, r0, eps, L)
        d = (log_p(th1) - 0.5 * r1 @ r1) - (log_p(th) - 0.5 * r0 @ r0)   # -ΔH
        dH.append(-d)
        if np.log(rng.random()) < d:
            th, acc = th1, acc + 1
        out[t] = th
    return out, acc / n, np.array(dH)

for eps in (0.1, 0.3, 0.5):
    s, a, dH = hmc([3.0, -3.0], eps, 20, 2000, np.random.default_rng(2))
    print(f"eps {eps}: 수용률 {a:.2f}  |ΔH| 중앙값 {np.median(np.abs(dH[500:])):8.2g}  표준편차 {s[500:].std(0).round(2)}")

# %% cell 4 — 수렴 진단: split R-hat
def split_rhat(chains):                                 # (M, N) → 스칼라
    h = chains.shape[1] // 2
    c = np.vstack([chains[:, :h], chains[:, h:2 * h]])  # (2M, N/2) 반으로 나눈 체인
    n = c.shape[1]
    W = c.var(axis=1, ddof=1).mean()                    # 체인 내 분산
    B = n * c.mean(axis=1).var(ddof=1)                  # 체인 간 분산
    return np.sqrt(((n - 1) / n * W + B / n) / W)

starts = [[4, 4], [-4, -4], [4, -4], [-4, 4]]           # 흩어진 시작점
rng = np.random.default_rng(3)
for step, n in [(0.05, 1000), (0.5, 1000), (0.5, 10000)]:
    ch = np.array([metropolis(log_p, s0, step, n, rng)[0][n // 2:, 0] for s0 in starts])
    print(f"MH 보폭 {step:4}, 반복 {n:5}: R-hat {split_rhat(ch):.3f}")

# %% cell 5 — 유효 표본 크기(ESS)
def ess(chains):                                        # (M, N) → 스칼라
    M, N = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, 2 * N, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :N]    # (M, N) 자기공분산
    rho_t = (ac / ac[:, :1]).mean(axis=0)               # 체인 평균 자기상관
    tau, k = -1.0, 0
    while 2 * k + 1 < N and rho_t[2 * k] + rho_t[2 * k + 1] > 0:
        tau += 2 * (rho_t[2 * k] + rho_t[2 * k + 1])    # Geyer 초기 양수열
        k += 1
    return M * N / tau

rng = np.random.default_rng(4)
mh = np.array([metropolis(log_p, s0, 0.5, 2000, rng)[0][1000:, 0] for s0 in starts])
hm = np.array([hmc(s0, 0.3, 20, 2000, rng)[0][1000:, 0] for s0 in starts])
for name, ch in [("MH  (보폭 0.5)", mh), ("HMC (eps 0.3)", hm)]:
    print(f"{name}: 표본 {ch.size}개  R-hat {split_rhat(ch):.3f}  ESS {ess(ch):6.0f}")
