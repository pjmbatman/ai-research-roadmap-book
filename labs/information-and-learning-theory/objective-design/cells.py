"""14.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/information-and-learning-theory/objective-design/cells.py
"""
# %% cell 1 — 측정 오차가 다른 데이터: 우도가 정하는 가중치
import numpy as np

rng = np.random.default_rng(0)
x = np.linspace(0, 1, 40)                          # (N,)
sigma = np.where(np.arange(40) % 2 == 0, 0.05, 1.0)   # 정밀 장비와 거친 장비가 번갈아 측정, (N,)

def fit_slope(y, weights):                         # Σ w_i (y_i - a x_i)² 최소화의 닫힌 해
    return np.sum(weights * x * y) / np.sum(weights * x * x)

est = {"MSE (가중치 없음)": [], "가우시안 NLL (1/σ² 가중치)": []}
for _ in range(2000):
    y = 2.0 * x + sigma * rng.normal(size=40)      # 참 기울기 2
    est["MSE (가중치 없음)"].append(fit_slope(y, np.ones(40)))
    est["가우시안 NLL (1/σ² 가중치)"].append(fit_slope(y, 1 / sigma**2))
for name, a in est.items():
    print(f"{name:22s} 기울기 평균 {np.mean(a):.3f}   표준편차 {np.std(a):.3f}")

# %% cell 2 — MLE = forward KL 최소화
p = np.array([0.5, 0.3, 0.2])                      # 데이터 분포
q = np.array([0.4, 0.4, 0.2])                      # 모델
H_p = -np.sum(p * np.log(p))
KL = np.sum(p * np.log(p / q))
print(f"H(p) = {H_p:.4f}   KL(p||q) = {KL:.4f}   H(p) + KL = {H_p + KL:.4f}")
rng = np.random.default_rng(1)
for n in (100, 10_000, 1_000_000):
    xs = rng.choice(3, size=n, p=p)                # 데이터 (N,)
    nll_q = -np.mean(np.log(q[xs]))                # 모델의 평균 NLL
    nll_p = -np.mean(np.log(p[xs]))                # 참 분포의 평균 NLL
    print(f"N = {n:9,d}   평균 NLL(q) = {nll_q:.4f}   NLL(q) - NLL(p) = {nll_q - nll_p:.4f}")

# %% cell 3 — ELBO = log p(x) - KL(q || p(z|x))
prior = np.array([0.3, 0.7])                       # p(z), z ∈ {0, 1}
mu = np.array([-1.0, 2.0])                         # p(x | z) = N(mu_z, 1)
x_obs = 0.0
log_joint = np.log(prior) - 0.5 * (x_obs - mu) ** 2 - 0.5 * np.log(2 * np.pi)   # log p(x, z), (2,)
log_px = np.log(np.sum(np.exp(log_joint)))         # log p(x)
post = np.exp(log_joint - log_px)                  # p(z | x), (2,)

print(f"log p(x) = {log_px:.4f}   사후분포 p(z|x) = {post.round(3)}")
for name, qz in (("q = 사전분포", prior), ("q = 균등", np.array([0.5, 0.5])),
                 ("q = 사후분포", post), ("q = [0.95, 0.05]", np.array([0.95, 0.05]))):
    elbo = np.sum(qz * (log_joint - np.log(qz)))   # E_q[log p(x,z) - log q(z)]
    kl = np.sum(qz * np.log(qz / post))
    print(f"{name:16s} ELBO = {elbo:.4f}   KL(q||p(z|x)) = {kl:.4f}   합 = {elbo + kl:.4f}")

# %% cell 4 — 단위가 다른 두 손실 더하기
rng = np.random.default_rng(2)
x = rng.uniform(0, 1, size=50)                     # 공통 입력, (N,)
dT_k = 10.0 * 1.0 * x + rng.normal(0, 0.5, 50)     # 과제 A: 온도 상승 [K], 잡음 0.5 K
dp_pa = 5000.0 * 1.1 * x + rng.normal(0, 200, 50)  # 과제 B: 압력 상승 [Pa], 잡음 200 Pa

def shared_w(yA, a, wA, yB, b, wB):                # min wA Σ(yA - a w x)² + wB Σ(yB - b w x)²
    return (wA * a * x @ yA + wB * b * x @ yB) / ((wA * a**2 + wB * b**2) * (x @ x))

def learned_sigma(yA, a, yB, b, iters=50):         # 불확실성 가중치: σ_k와 w를 번갈아 최적화
    vA = vB = 1.0
    for _ in range(iters):
        w = shared_w(yA, a, 1 / vA, yB, b, 1 / vB)
        vA, vB = np.mean((yA - a * w * x) ** 2), np.mean((yB - b * w * x) ** 2)
    return w

for unit, s in (("Pa ", 1.0), ("kPa", 1e-3)):
    yB, b = dp_pa * s, 5000.0 * s
    print(f"압력 단위 {unit}  단순 합 w = {shared_w(dT_k, 10, 1, yB, b, 1):.3f}   "
          f"1/σ² 가중 w = {shared_w(dT_k, 10, 1 / 0.5**2, yB, b, 1 / (200 * s) ** 2):.3f}   "
          f"학습한 σ w = {learned_sigma(dT_k, 10, yB, b):.3f}")

# %% cell 5 — 집계 방식: padding을 포함한 평균과 mask 평균
rng = np.random.default_rng(3)
lengths = np.array([10, 4, 2])                     # 시퀀스 길이, (B,)
raw = rng.uniform(0.5, 1.5, size=(3, 10))          # token별 손실(가장 긴 길이까지), (B, 10)
for T in (10, 20):                                 # batch를 padding한 길이
    mask = np.arange(T)[None, :] < lengths[:, None]               # (B, T)
    loss = np.where(mask, np.pad(raw, ((0, 0), (0, T - 10))), 0.0)  # padding 위치의 손실은 0, (B, T)
    padded = loss.mean()                           # padding까지 평균
    token = loss.sum() / mask.sum()                # 실제 token에 대한 평균
    seq = np.mean(loss.sum(1) / lengths)           # 시퀀스별 평균의 평균
    print(f"T = {T:2d}  padding 포함 평균 {padded:.3f}   token 평균 {token:.3f}   시퀀스 평균의 평균 {seq:.3f}")
