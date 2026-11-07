"""21.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reliability/uncertainty-quantification/cells.py
"""
# %% cell 1 — 부트스트랩 앙상블로 aleatoric·epistemic 분해
import numpy as np

rng = np.random.default_rng(0)
n = 300
x = np.concatenate([rng.uniform(-3, -1, n // 2), rng.uniform(1, 3, n // 2)])  # (N,) [-1, 1]은 비어 있다
sd_true = lambda x: 0.05 + 0.1 * (x + 3)             # 참 잡음 sd: 오른쪽으로 커진다
y = np.sin(x) + rng.normal(size=n) * sd_true(x)      # (N,)

def fit_member(xb, yb):
    mu_c = np.polyfit(xb, yb, 5)                     # 평균: 5차 다항식
    r2 = (yb - np.polyval(mu_c, xb)) ** 2
    lv_c = np.polyfit(xb, np.log(r2) + 1.27, 2)      # log 분산: 2차 다항식 (E[log χ²₁] ≈ -1.27 보정)
    return mu_c, lv_c

M = 20
members = [fit_member(x[i], y[i]) for i in rng.integers(0, n, size=(M, n))]  # 부트스트랩 재표본 M개

x0 = np.array([-2.0, 2.0, 0.0, 4.0])
mu = np.array([np.polyval(a, x0) for a, _ in members])           # (M, 4) 구성원별 평균
var = np.array([np.exp(np.polyval(b, x0)) for _, b in members])  # (M, 4) 구성원별 분산
alea, epi = var.mean(axis=0), mu.var(axis=0)                     # 전분산 법칙의 두 항
for xi, a, e in zip(x0, alea, epi):
    print(f"x = {xi:4.1f}  aleatoric sd {np.sqrt(a):.3f}  epistemic sd {np.sqrt(e):.3f}  참 잡음 sd {sd_true(xi):.3f}")

# %% cell 2 — 앙상블 크기와 epistemic 추정의 흔들림
def epistemic_sd(M, x_eval):
    idx = rng.integers(0, n, size=(M, n))
    mus = [np.polyval(np.polyfit(x[i], y[i], 5), x_eval) for i in idx]
    return np.std(mus)

R = 100
for M in (2, 5, 20):
    est = np.array([epistemic_sd(M, 4.0) for _ in range(R)])   # (R,) 같은 데이터, 다른 앙상블
    print(f"M = {M:2d}  epistemic sd 평균 {est.mean():.2f}  반복 간 sd {est.std():.2f}  상대 흔들림 {est.std() / est.mean():.0%}")

# %% cell 3 — softmax 확률은 학습 범위를 모른다
def fit_logreg(X, t, steps=2000, lr=0.5):
    Xb = np.hstack([X, np.ones((len(X), 1))])          # (N, 3) bias 열 추가
    w = np.zeros(3)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - t) / len(t)              # 평균 교차 엔트로피의 gradient
    return w

Xc = np.vstack([rng.normal([-1, 0], 1.0, (200, 2)), rng.normal([1, 0], 1.0, (200, 2))])  # (400, 2)
tc = np.repeat([0.0, 1.0], 200)                                                         # (400,)
ens = [fit_logreg(Xc[i], tc[i]) for i in rng.integers(0, 400, size=(10, 400))]          # 부트스트랩 10개

queries = {"경계 근처 (0, 0)": [0.0, 0.0], "학습 범위 안 (1, 0)": [1.0, 0.0], "멀리 밖 (20, 30)": [20.0, 30.0]}
for name, q in queries.items():
    p = np.array([1 / (1 + np.exp(-(w[:2] @ q + w[2]))) for w in ens])
    print(f"{name:14s}  p(class 1) 평균 {p.mean():.3f}  구성원 간 sd {p.std():.3f}")

# %% cell 4 — 입력 불확실성의 Monte Carlo 전파
R_J_PER_MOL_K = 8.314
temp_k = 300.0
ea_j_per_mol = rng.normal(80e3, 2e3, size=100_000)          # (S,) 활성화 에너지의 불확실성
ratio = np.exp(-(ea_j_per_mol - 80e3) / (R_J_PER_MOL_K * temp_k))   # (S,) k / k(평균 Ea)

sd_lin = 2e3 / (R_J_PER_MOL_K * temp_k)                      # 1차 선형화: d ln k = -dEa / (R T)
lo, med, hi = np.percentile(ratio, [2.5, 50, 97.5])
print(f"Monte Carlo  평균 {ratio.mean():.2f}  중앙값 {med:.2f}  95% 구간 [{lo:.2f}, {hi:.2f}]")
print(f"선형화       평균 1.00  sd {sd_lin:.2f}  95% 구간 [{1 - 1.96 * sd_lin:.2f}, {1 + 1.96 * sd_lin:.2f}]")
