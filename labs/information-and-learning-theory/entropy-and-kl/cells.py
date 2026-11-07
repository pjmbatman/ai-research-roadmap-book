"""14.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/information-and-learning-theory/entropy-and-kl/cells.py
"""
# %% cell 1 — 엔트로피와 부호 길이
import numpy as np

def entropy_bits(p):
    p = p[p > 0]                                   # 0 log 0 = 0
    return -np.sum(p * np.log2(p))

dists = {
    "균등 8개": np.full(8, 1 / 8),
    "2의 거듭제곱": np.array([0.5, 0.25, 0.125, 0.125]),
    "일반": np.array([0.4, 0.3, 0.2, 0.1]),
}
for name, p in dists.items():
    code_len = np.ceil(-np.log2(p))                # Shannon 부호: 결과마다 ⌈-log2 p⌉ 비트
    L = np.sum(p * code_len)
    print(f"H = {entropy_bits(p):.3f} bit   평균 부호 길이 L = {L:.3f} bit   ({name})")

# %% cell 2 — 미분 엔트로피는 단위에 따라 바뀐다
rng = np.random.default_rng(0)
x_m = rng.normal(0.0, 0.002, size=200_000)         # 위치 측정 잡음 [m], σ = 2 mm

def gauss_h_nats(sigma):                           # N(μ, σ²)의 미분 엔트로피 ½ log(2πeσ²)
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

for unit, scale in (("m", 1.0), ("mm", 1e3)):
    x = x_m * scale
    mc = -np.mean(-0.5 * np.log(2 * np.pi * x.std() ** 2) - x**2 / (2 * x.std() ** 2))
    print(f"단위 {unit:2s}  h = {gauss_h_nats(x.std()):7.3f} nat   (Monte Carlo {mc:7.3f})")
print(f"차이 = log(1000) = {np.log(1000):.3f} nat")

# %% cell 3 — 교차엔트로피 = 엔트로피 + KL, 그리고 비대칭성
def kl(p, q):
    m = p > 0                                      # 0 log(0/q) = 0
    if np.any(q[m] == 0):
        return np.inf                              # p > 0인데 q = 0인 곳이 있으면 무한대
    return np.sum(p[m] * np.log(p[m] / q[m]))      # nat

def cross_entropy(p, q):
    m = p > 0
    return -np.sum(p[m] * np.log(q[m]))

p = np.array([0.70, 0.20, 0.10])
q = np.array([0.40, 0.40, 0.20])
H_p = cross_entropy(p, p)
print(f"H(p) = {H_p:.4f}   H(p,q) = {cross_entropy(p, q):.4f}   H(p) + KL(p||q) = {H_p + kl(p, q):.4f}")
print(f"KL(p||q) = {kl(p, q):.4f}   KL(q||p) = {kl(q, p):.4f}")
r = np.array([0.5, 0.5, 0.0])                      # 세 번째 결과를 불가능하다고 보는 모델
print(f"KL(p||r) = {kl(p, r)}   KL(r||p) = {kl(r, p):.4f}")

# %% cell 4 — 쌍봉 분포에 가우시안 하나를 맞추기: forward KL vs reverse KL
x = np.linspace(-8, 8, 4001)                       # 적분 격자, (G,)
dx = x[1] - x[0]
def normal_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

p = 0.5 * normal_pdf(x, -2.0, 0.6) + 0.5 * normal_pdf(x, 2.0, 0.6)   # 참 분포, (G,)
mus, sigmas = np.linspace(-3, 3, 121), np.linspace(0.2, 3.0, 57)
best = {"forward KL(p||q)": (np.inf,), "reverse KL(q||p)": (np.inf,)}
for mu in mus:
    for s in sigmas:
        q = normal_pdf(x, mu, s)
        fwd = np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300))) * dx
        rev = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
        for key, val in (("forward KL(p||q)", fwd), ("reverse KL(q||p)", rev)):
            if val < best[key][0]:
                best[key] = (val, mu, s)
for key, (val, mu, s) in best.items():
    print(f"{key}  최소 {val:.3f} nat  →  q = N({mu:+.2f}, {s:.2f}²)")
print(f"p의 평균 {np.sum(x * p) * dx:+.2f}, 표준편차 {np.sqrt(np.sum(x**2 * p) * dx):.2f}")

# %% cell 5 — 겹치지 않는 두 분포: KL, JS, Wasserstein
from scipy.stats import wasserstein_distance

support = np.arange(10)                            # 위치 0..9
p = np.zeros(10); p[[1, 2]] = 0.5                  # 질량이 1, 2에 있다
for shift in (2, 4, 6):
    q = np.roll(p, shift)                          # 같은 모양을 shift칸 옮긴다
    m = 0.5 * (p + q)
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    w1 = wasserstein_distance(support, support, p, q)
    print(f"이동 {shift}칸  KL(p||q) = {kl(p, q)}   JS = {js:.4f} (log 2 = {np.log(2):.4f})   W1 = {w1:.1f}")
