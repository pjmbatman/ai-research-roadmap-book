"""14.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/information-and-learning-theory/mutual-information/cells.py
"""
# %% cell 1 — 결합분포 표에서 상호정보량 계산
import numpy as np

def H(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))                  # nat

def mutual_info(pxy):                              # pxy: 결합확률 표, (|X|, |Y|)
    return H(pxy.sum(1)) + H(pxy.sum(0)) - H(pxy.ravel())

xs = np.array([-1, 0, 1])                          # X는 세 값 중 하나, 같은 확률
ys = np.array([0, 1])
pxy = np.array([[1 / 3 * (y == x**2) for y in ys] for x in xs])   # Y = X², (3, 2)
EX = np.sum(pxy * xs[:, None]); EY = np.sum(pxy * ys[None, :])
cov = np.sum(pxy * xs[:, None] * ys[None, :]) - EX * EY
print(f"Y = X²       Cov(X, Y) = {abs(cov):.3f}   I(X; Y) = {mutual_info(pxy):.4f} nat   H(Y) = {H(pxy.sum(0)):.4f}")
indep = np.outer([0.2, 0.3, 0.5], [0.6, 0.4])      # 독립: p(x, y) = p(x) p(y)
print(f"독립         I(X; Y) = {abs(mutual_info(indep)):.4f} nat")

# %% cell 2 — data processing inequality: 처리하면 정보는 늘지 않는다
rng = np.random.default_rng(0)
px = np.full(4, 0.25)                              # X: 네 가지 상태, (4,)
noisy = 0.7 * np.eye(4) + 0.1                      # p(y | x): 70% 정답 + 잡음, 행 합 1, (4, 4)
pxy = px[:, None] * noisy                          # p(x, y), (4, 4)
merge = np.array([[1, 0], [1, 0], [0, 1], [0, 1]]) # Z = g(Y): 네 상태를 두 묶음으로, (4, 2)
pxz = pxy @ merge                                  # p(x, z), (4, 2)
rand = rng.dirichlet(np.ones(3), size=4)           # 임의의 확률적 처리 p(w | y), (4, 3)
pxw = pxy @ rand                                   # p(x, w), (4, 3)
print(f"I(X; Y) = {mutual_info(pxy):.4f}")
print(f"I(X; Z) = {mutual_info(pxz):.4f}   (Z = Y를 두 묶음으로 합침)")
print(f"I(X; W) = {mutual_info(pxw):.4f}   (W = Y에 무작위 채널 적용)")

# %% cell 3 — 히스토그램 추정량의 편향: bin 수와 표본 수
def mi_hist(x, y, bins):
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    return mutual_info(counts / counts.sum())

rho = 0.6
true_mi = -0.5 * np.log(1 - rho**2)                # 이변량 가우시안의 MI [nat]
print(f"참값: 상관 0.6 → {true_mi:.3f} nat,  독립 → 0")
print("   N   bins   상관 0.6    독립")
for n in (200, 2000):
    for bins in (5, 20, 50):
        est = []
        for seed in range(20):
            r = np.random.default_rng(seed)
            x, e = r.normal(size=(2, n))           # (N,), (N,)
            y = rho * x + np.sqrt(1 - rho**2) * e
            est.append((mi_hist(x, y, bins), mi_hist(x, e, bins)))
        m = np.mean(est, axis=0)
        print(f"{n:5d}  {bins:4d}   {m[0]:.3f}      {m[1]:.3f}")

# %% cell 4 — InfoNCE 하한은 log K를 넘지 못한다
rho, d = 0.9, 10                                   # 성분마다 상관 0.9인 10차원 쌍
true_mi = -0.5 * d * np.log(1 - rho**2)            # 성분별 MI의 합

def critic(x, y):                                  # 최적 critic log p(y|x) - log p(y) (상수 제외), (K, K)
    sq = (y**2).sum(1)[None, :] - 2 * rho * x @ y.T + rho**2 * (x**2).sum(1)[:, None]   # ‖y_j - ρx_i‖²
    return -sq / (2 * (1 - rho**2)) + (y**2).sum(1)[None, :] / 2

r = np.random.default_rng(1)
print(f"참 MI = {true_mi:.3f} nat")
for K in (4, 16, 64, 256, 1024):
    vals = []
    for _ in range(10):
        x = r.normal(size=(K, d))                  # (K, d)
        y = rho * x + np.sqrt(1 - rho**2) * r.normal(size=(K, d))
        S = critic(x, y)                           # S[i, j] = f(x_i, y_j)
        lse = np.log(np.mean(np.exp(S - S.max(1, keepdims=True)), axis=1)) + S.max(1)
        vals.append(np.mean(np.diag(S) - lse))     # mean_i [f(x_i,y_i) - log(1/K Σ_j e^f(x_i,y_j))]
    print(f"K = {K:4d}   InfoNCE = {np.mean(vals):.3f}   log K = {np.log(K):.3f}")
