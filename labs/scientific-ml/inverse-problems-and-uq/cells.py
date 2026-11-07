"""24.7 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

모든 셀은 NumPy만 쓰는 브라우저 셀이다(exec="browser", session="inverse").
실행: uv run --project labs python labs/scientific-ml/inverse-problems-and-uq/cells.py
"""
# %% cell 1 — 흐림 복원과 Tikhonov 정규화
import numpy as np

n = 80
s = np.linspace(0, 1, n)                                     # (n,) 위치
A = np.exp(-0.5 * (s[:, None] - s[None, :])**2 / 0.03**2)
A /= A.sum(axis=1, keepdims=True)                            # (n, n) 흐림 연산자: 이웃의 가중 평균
x_true = (np.abs(s - 0.3) < 0.1) + 0.5 * np.exp(-0.5 * (s - 0.7)**2 / 0.05**2)   # 참 신호 (n,)
rng = np.random.default_rng(0)
sigma = 0.01                                                 # 측정 잡음 표준편차
y = A @ x_true + sigma * rng.normal(size=n)                  # 측정 (n,)

def tikhonov(A, y, lam):                                     # argmin ||A x - y||^2 + lam ||x||^2
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)

sv = np.linalg.svd(A, compute_uv=False)
print(f"특이값 최대 {sv[0]:.2f}, 1e-6 미만인 것 {np.sum(sv < 1e-6)}개 / {n}")
print("정규화 없이 풀면 오차 > 1e3:", np.linalg.norm(np.linalg.lstsq(A, y, rcond=None)[0] - x_true) > 1e3)
for lam in [1e-6, 1e-4, 1e-2, 1e0]:
    x = tikhonov(A, y, lam)
    print(f"lam = {lam:.0e}: 잔차 {np.linalg.norm(A @ x - y):.3f}  해 노름 {np.linalg.norm(x):5.2f}"
          f"  오차 {np.linalg.norm(x - x_true):5.2f}")

# %% cell 2 — L-curve와 불일치 원리로 lam 고르기
lams = np.logspace(-8, 1, 91)
xs = [tikhonov(A, y, l) for l in lams]
res = np.array([np.linalg.norm(A @ x - y) for x in xs])
sol = np.array([np.linalg.norm(x) for x in xs])
err = np.array([np.linalg.norm(x - x_true) for x in xs])
r, e = np.log(res), np.log(sol)                              # L-curve: (log 잔차, log 해 노름)
dr, de = np.gradient(r), np.gradient(e)
curv = (dr * np.gradient(de) - de * np.gradient(dr)) / (dr**2 + de**2)**1.5
i_corner = np.argmax(curv[5:-5]) + 5                         # 양 끝의 수치 잡음 제외
i_disc = np.argmin(np.abs(res - np.sqrt(n) * sigma))         # 불일치 원리: 잔차 ≈ 잡음 크기
for name, i in [("L-curve 모서리  ", i_corner), ("불일치 원리     ", i_disc), ("오차 최소(참값 필요)", np.argmin(err))]:
    print(f"{name}  lam = {lams[i]:.1e}  오차 {err[i]:.2f}")

# %% cell 3 — 선형 가우시안 베이지안 역문제
tau = 0.5                                                    # 사전분포 x ~ N(0, tau^2 I)
Sigma_post = np.linalg.inv(A.T @ A / sigma**2 + np.eye(n) / tau**2)   # (n, n) 사후 공분산
mu_post = Sigma_post @ A.T @ y / sigma**2                    # (n,) 사후 평균
sd_post = np.sqrt(np.diag(Sigma_post))
print("사후 평균 = Tikhonov(lam = sigma^2 / tau^2):", np.allclose(mu_post, tikhonov(A, y, sigma**2 / tau**2)))
print(f"점마다의 사후 표준편차 {sd_post.min():.3f} – {sd_post.max():.3f} (사전 {tau})")
print(f"95% 구간이 참값을 포함한 위치 비율 {np.mean(np.abs(mu_post - x_true) < 1.96 * sd_post):.2f}")
w = (np.abs(s - 0.3) < 0.1) / np.sum(np.abs(s - 0.3) < 0.1)   # 구간 [0.2, 0.4]의 평균을 내는 가중치 (n,)
print(f"구간 평균: 사후 {w @ mu_post:.3f} ± {np.sqrt(w @ Sigma_post @ w):.3f}  참값 {w @ x_true:.3f}")

# %% cell 4 — 식별 불가능성: MCMC로 본 사후분포와 예측
def rate(s_mm, vmax, km):                                    # Michaelis–Menten 반응 속도
    return vmax * s_mm / (km + s_mm)

def mcmc(s_mm, v_obs, noise=0.005, n_chain=16, n_step=3000, seed=0):   # 로그 파라미터의 random-walk Metropolis
    r = np.random.default_rng(seed)
    def logpost(th):                                         # th: (C, 2) = (log vmax, log km), 사전 N(0, 3^2)
        pred = rate(s_mm[None, :], np.exp(th[:, :1]), np.exp(th[:, 1:]))
        return -0.5 * np.sum((v_obs - pred)**2, axis=1) / noise**2 - 0.5 * np.sum(th**2, axis=1) / 3**2
    th = r.normal(0, 1, size=(n_chain, 2)); lp = logpost(th)
    out = []
    for i in range(n_step):
        prop = th + 0.1 * r.normal(size=th.shape)
        lp_prop = logpost(prop)
        acc = np.log(r.uniform(size=n_chain)) < lp_prop - lp
        th[acc], lp[acc] = prop[acc], lp_prop[acc]
        if i >= 1000:                                        # 앞 1000 step은 버린다
            out.append(th.copy())
    return np.exp(np.concatenate(out))                       # (n_chain * 2000, 2) = (vmax, km)

rng = np.random.default_rng(1)
q = lambda a, d: "–".join(f"{v:.{d}f}" for v in np.percentile(a, [5, 95]))   # 90% 구간 문자열
for name, s_mm in [("저농도만 0.1–0.5 mM", np.linspace(0.1, 0.5, 10)), ("넓게 0.1–20 mM    ", np.geomspace(0.1, 20, 10))]:
    v_obs = rate(s_mm, 2.0, 5.0) + 0.005 * rng.normal(size=10)   # 참값 vmax = 2, km = 5 mM
    post = mcmc(s_mm, v_obs)
    print(f"{name}: vmax {q(post[:, 0], 2)}  km {q(post[:, 1], 1)}  vmax/km {q(post[:, 0] / post[:, 1], 3)}"
          f"  v(20 mM) {q(rate(20.0, post[:, 0], post[:, 1]), 2)}  log 상관 {np.corrcoef(np.log(post.T))[0, 1]:.2f}")

# %% cell 5 — 국소 민감도와 Sobol 지수
def ishigami(X, a=7.0, b=0.1):                               # X: (N, 3), 각 성분 U(-pi, pi)
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2 + b * X[:, 2]**4 * np.sin(X[:, 0])

x0, h = np.zeros((1, 3)), 1e-5
grad = [(ishigami(x0 + h * np.eye(3)[[i]]) - ishigami(x0 - h * np.eye(3)[[i]]))[0] / (2 * h) for i in range(3)]
print("원점에서의 국소 민감도 df/dx_i:", ", ".join(f"{g:.3f}" for g in grad))
rng = np.random.default_rng(0)
N = 100_000
A_s, B_s = rng.uniform(-np.pi, np.pi, (N, 3)), rng.uniform(-np.pi, np.pi, (N, 3))
fA, fB = ishigami(A_s), ishigami(B_s)
V = np.var(np.concatenate([fA, fB]))                         # 출력의 총분산
for i in range(3):
    ABi = A_s.copy(); ABi[:, i] = B_s[:, i]                  # A에서 i번째 열만 B로 바꾼 표본
    fABi = ishigami(ABi)
    S1 = max(np.mean(fB * (fABi - fA)) / V, 0.0)             # 1차 지수, 추정 잡음으로 생긴 음수는 0으로
    ST = 0.5 * np.mean((fA - fABi)**2) / V                   # 총 지수
    print(f"x{i + 1}: S1 = {S1:.2f}   ST = {ST:.2f}")
