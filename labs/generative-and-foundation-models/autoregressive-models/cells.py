"""19.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/generative-and-foundation-models/autoregressive-models/cells.py
"""
# %% cell 1 — chain rule: 어떤 순서로 분해해도 결합분포가 정확히 복원된다
import numpy as np
from itertools import product

rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(8)).reshape(2, 2, 2)    # (2, 2, 2) 이진 변수 3개의 결합분포

def chain_rule_prob(x, order):
    """p(x) = prod_i p(x_{o_i} | x_{o_1}, ..., x_{o_{i-1}})"""
    prob, fixed = 1.0, {}
    for i in order:
        def marg(assign):                         # 고정된 변수만 맞추고 나머지는 더한다
            return sum(P[s] for s in product([0, 1], repeat=3)
                       if all(s[j] == v for j, v in assign.items()))
        prob *= marg({**fixed, i: x[i]}) / marg(fixed)
        fixed[i] = x[i]
    return prob

for order in [(0, 1, 2), (2, 0, 1)]:
    err = max(abs(chain_rule_prob(x, order) - P[x]) for x in product([0, 1], repeat=3))
    print(f"순서 {order}: 최대 오차 < 1e-12: {err < 1e-12}")
x = (1, 0, 1)
terms = [P[1].sum(), P[1, 0].sum() / P[1].sum(), P[1, 0, 1] / P[1, 0].sum()]
print("log p(x) =", np.round(np.log(P[x]), 4), "= 조건부 로그의 합", np.round(np.sum(np.log(terms)), 4))

# %% cell 2 — causal mask: 출력 t는 입력 t 이전에만 의존한다
d = 5
mask = np.tril(np.ones((d, d)), k=-1)             # (d, d) 대각선 아래만 1: t는 0..t-1만 본다
W = rng.normal(size=(d, d)) * mask                # (d, d) 가려진 가중치

def logits(x):                                    # x: (d,) -> (d,) 모든 위치의 조건부 logit을 한 번에
    return W @ np.tanh(x) + 0.1

x0 = rng.normal(size=d)
J = np.zeros((d, d))                              # J[t, s] = d logit_t / d x_s (유한차분)
for s in range(d):
    e = np.zeros(d); e[s] = 1e-6
    J[:, s] = (logits(x0 + e) - logits(x0)) / 1e-6
print("출력 t가 입력 s >= t에 의존하지 않음:", np.all(np.abs(np.triu(J)) < 1e-12))
print("의존 패턴 (1 = 의존):")
print((np.abs(J) > 1e-8).astype(int))

# %% cell 3 — 샘플링 전략이 바꾸는 분포
def filtered(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)                        # 확률이 큰 순서
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:                         # 누적확률이 top_p에 처음 도달할 때까지
        n_keep = np.searchsorted(np.cumsum(p[order]), top_p) + 1
        keep[order[n_keep:]] = False
    p = np.where(keep, p, 0.0)
    return p / p.sum()

lg = np.log(np.array([0.40, 0.20, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01, 0.01]))
def entropy(p):
    return -np.sum(p[p > 0] * np.log(p[p > 0]))
for name, kw in [("T=1", {}), ("T=0.5", {"temperature": 0.5}), ("T=2", {"temperature": 2.0}),
                 ("top-k=3", {"top_k": 3}), ("top-p=0.9", {"top_p": 0.9})]:
    p = filtered(lg, **kw)
    print(f"{name:10s} 후보 {np.sum(p > 0):2d}개  최고 확률 {p.max():.3f}  엔트로피 {entropy(p):.3f}")

# %% cell 4 — 연속값 자기회귀: 가우시안 head와 정확한 log-likelihood
def simulate(n, R, rng, noise=0.02):              # 참 과정: 진폭이 스스로 제한되는 비선형 진동
    x = np.zeros((R, n)); x[:, 0], x[:, 1] = 1.0, 0.95
    for t in range(2, n):
        a, b = x[:, t - 1], x[:, t - 2]
        x[:, t] = 1.9 * a - 0.95 * b + 0.15 * (a - a**3) + noise * rng.normal(size=R)
    return x                                      # (R, n)

def feats(a, b, cubic):                           # 입력: 직전 두 값 (teacher forcing)
    return np.stack([a, b] + ([a**3] if cubic else []), axis=-1)

x_tr, x_te = simulate(3000, 1, rng)[0], simulate(1000, 1, rng)[0]
models = {}
for name, cubic in [("선형 AR(2)", False), ("AR(2)+세제곱", True)]:
    w = np.linalg.lstsq(feats(x_tr[1:-1], x_tr[:-2], cubic), x_tr[2:], rcond=None)[0]
    sig = (x_tr[2:] - feats(x_tr[1:-1], x_tr[:-2], cubic) @ w).std()   # head의 표준편차
    r = x_te[2:] - feats(x_te[1:-1], x_te[:-2], cubic) @ w
    nll = np.mean(0.5 * r**2 / sig**2 + 0.5 * np.log(2 * np.pi * sig**2))
    models[name] = (w, sig, cubic)
    print(f"{name:12s} 계수 {np.round(w, 3)}  1-step RMSE {np.sqrt(np.mean(r**2)):.4f}  test NLL/step {nll:.3f}")

# %% cell 5 — 자유 생성: teacher forcing 평가가 놓치는 것과 온도
def generate(w, sig, cubic, R=200, H=400, tau=1.0, seed=3):
    g = np.random.default_rng(seed)
    xs = np.zeros((R, H)); xs[:, 0], xs[:, 1] = 1.0, 0.95
    for t in range(2, H):                         # 자기 출력을 다음 입력으로 쓴다
        xs[:, t] = feats(xs[:, t - 1], xs[:, t - 2], cubic) @ w + tau * sig * g.normal(size=R)
    return xs                                     # (R, H)

q95 = lambda xs: np.quantile(np.abs(xs[:, 200:]), 0.95)
print(f"참 과정        |x|의 95% 분위수 {q95(simulate(400, 200, np.random.default_rng(3))):.3f}")
for name, (w, sig, cubic) in models.items():
    print(f"{name:12s}  |x|의 95% 분위수 {q95(generate(w, sig, cubic)):.3f}")
w, sig, cubic = models["AR(2)+세제곱"]
for tau in [0.5, 1.0, 2.0]:
    xs = generate(w, sig, cubic, tau=tau)
    print(f"온도 {tau}: t=100에서 경로 간 표준편차 {xs[:, 100].std():.3f}  |x| 95% 분위수 {q95(xs):.3f}")
