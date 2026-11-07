"""20.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reinforcement-learning/bandits-and-offline-rl/cells.py
"""
# %% cell 1 — Bernoulli 밴딧: ε-greedy, UCB, Thompson sampling의 후회
import numpy as np
p_true = np.array([0.30, 0.45, 0.50, 0.55, 0.60])   # 팔 5개의 성공 확률(알고리즘은 모른다)
K, T, RUNS = p_true.size, 2000, 200
def bandit(policy, seed=0):
    rng = np.random.default_rng(seed)
    n, s = np.zeros((RUNS, K)), np.zeros((RUNS, K))  # 팔별 시행 수, 성공 수, (RUNS, K)
    regret, rows = np.zeros(T), np.arange(RUNS)
    for t in range(T):
        a = policy(n, s, t, rng)                     # (RUNS,) 실행마다 고른 팔
        r = rng.random(RUNS) < p_true[a]             # 보상 0/1
        n[rows, a] += 1; s[rows, a] += r
        regret[t] = np.mean(p_true.max() - p_true[a])  # 기대 후회(실행 평균)
    return regret.cumsum()
def eps_greedy(n, s, t, rng, eps=0.1):
    mean = np.where(n > 0, s / np.maximum(n, 1), 1.0)   # 안 해 본 팔은 먼저 해 본다
    explore = rng.random(len(n)) < eps
    return np.where(explore, rng.integers(K, size=len(n)), mean.argmax(axis=1))
def ucb1(n, s, t, rng):
    if t < K:
        return np.full(len(n), t)                    # 처음에 팔마다 한 번씩
    return (s / n + np.sqrt(2 * np.log(t) / n)).argmax(axis=1)
def thompson(n, s, t, rng):                          # Beta(1 + 성공, 1 + 실패)에서 하나씩 뽑는다
    return rng.beta(1 + s, 1 + n - s).argmax(axis=1)
for name, pol in [("무작위", lambda n, s, t, rng: rng.integers(K, size=len(n))),
                  ("ε-greedy(0.1)", eps_greedy), ("UCB1", ucb1), ("Thompson", thompson)]:
    R = bandit(pol)
    print(f"{name:14s} 누적 후회  t=200: {R[199]:6.1f}   t=1000: {R[999]:6.1f}   t=2000: {R[-1]:6.1f}"
          f"   (뒤 1000 step 증가 {R[-1] - R[999]:5.1f})")

# %% cell 2 — 실험 조건 선택: 예산 100번으로 합성 조건 8개 중 가장 좋은 것 찾기
p_cond = np.array([0.05, 0.10, 0.12, 0.15, 0.20, 0.22, 0.25, 0.40])   # 조건별 합성 성공률
def campaign(rule, rng, budget=100):
    n, s = np.zeros(8), np.zeros(8)
    for t in range(budget):
        a = int(rng.integers(8)) if rule == "random" else int(rng.beta(1 + s, 1 + n - s).argmax())
        n[a] += 1; s[a] += rng.random() < p_cond[a]
    best = int(((1 + s) / (2 + n)).argmax())         # 끝에서 사후 평균이 가장 큰 조건을 추천
    return s.sum(), best == 7

rng = np.random.default_rng(1)
for rule in ["random", "thompson"]:
    res = np.array([campaign(rule, rng) for _ in range(500)])          # 캠페인 500번 반복
    print(f"{rule:9s} 성공한 합성 수 평균 {res[:, 0].mean():5.1f}   최선 조건을 맞게 추천한 비율 {res[:, 1].mean():.2f}")

# %% cell 3 — 오프라인 데이터와 분포 이동: 탐욕 선택의 과대평가
mu = np.array([0.1, 0.2, 0.3, 0.4, 0.6,             # 행동 0–4: 데이터에 자주 나온다
               0.3, 0.5, 0.2, 0.65, 0.1])            # 행동 5–9: 데이터에 드물다, 참 평균 보상
b_prob = np.array([0.19] * 5 + [0.01] * 5)           # 데이터를 모은 행동 정책 (합 1)
rng = np.random.default_rng(2)
out = {"탐욕(평균 최대)": [], "보수적(하한 최대)": []}
for _ in range(300):                                 # 오프라인 데이터셋 300개
    a = rng.choice(10, size=300, p=b_prob)           # 행동 정책이 모은 300개, (300,)
    r = mu[a] + rng.normal(0, 0.5, size=300)         # 잡음 있는 보상
    n = np.bincount(a, minlength=10)                 # 행동별 표본 수, (10,)
    mean = np.bincount(a, weights=r, minlength=10) / np.maximum(n, 1)
    mean[n == 0] = -np.inf                           # 데이터에 없는 행동은 고르지 않는다
    lcb = mean - 1.0 / np.sqrt(np.maximum(n, 1))     # 표본이 적을수록 크게 깎은 하한
    for key, score in [("탐욕(평균 최대)", mean), ("보수적(하한 최대)", lcb)]:
        k = int(score.argmax())
        out[key].append((score[k], mu[k], n[k]))     # 선택 기준 값, 실제 가치, 표본 수
for key, v in out.items():
    v = np.array(v)
    print(f"{key:10s}  기준 값 평균 {v[:, 0].mean():.3f}  실제 가치 평균 {v[:, 1].mean():.3f}"
          f"  고른 행동의 표본 수 중앙값 {np.median(v[:, 2]):.0f}")
print(f"참 최적 행동의 가치 {mu.max():.3f}, 데이터에 충분한 행동 중 최선 {mu[:5].max():.3f}")

# %% cell 4 — 오프 정책 평가: importance sampling과 지평
def ope(H, n_traj=1000, seed=3):                     # 매 step 행동 2개, 보상 = 행동 1이면 1
    rng = np.random.default_rng(seed)
    pb, pe = np.array([0.5, 0.5]), np.array([0.2, 0.8])   # 행동 정책, 평가할 정책
    est_ois, est_wis, ess = [], [], []
    for _ in range(100):                             # 오프라인 데이터셋 100개로 분포를 본다
        a = rng.choice(2, size=(n_traj, H), p=pb)    # (N, H) 기록된 행동
        G = a.sum(axis=1).astype(float)              # 궤적별 이득(할인 없음)
        w = np.prod(pe[a] / pb[a], axis=1)           # 궤적별 importance weight, (N,)
        est_ois.append(np.mean(w * G))               # 보통(ordinary) IS
        est_wis.append(np.sum(w * G) / np.sum(w))    # 가중(self-normalized) IS
        ess.append(w.sum() ** 2 / np.sum(w**2) / n_traj)
    true = 0.8 * H
    for name, e in [("OIS", est_ois), ("WIS", est_wis)]:
        e = np.array(e)
        print(f"H={H:2d} {name}  참값 {true:4.1f}  추정 평균 {e.mean():5.2f}  표준편차 {e.std():5.2f}", end="")
        print(f"   ESS/N 중앙값 {np.median(ess):.3f}" if name == "WIS" else "")

for H in (1, 5, 20):
    ope(H)

# %% cell 5 — 안전 제약: 비용 예산이 있는 선택과 Lagrangian
rew = np.array([0.2, 0.5, 0.8, 0.9])                 # 조건별 기대 보상(예: 수율)
cost = np.array([0.0, 0.1, 0.4, 0.9])                # 조건별 기대 비용(예: 위험 사건 확률)
budget = 0.25                                        # 허용하는 평균 비용
print("제약 없는 최선:", int(rew.argmax()), " 비용", cost[rew.argmax()])
lo, hi = 0.0, 10.0
for _ in range(60):                                  # 쌍대 변수 λ를 이분법으로 찾는다
    lam = (lo + hi) / 2
    a = int(np.argmax(rew - lam * cost))             # λ가 주어지면 벌점 보상의 탐욕 선택
    lo, hi = (lam, hi) if cost[a] > budget else (lo, lam)
a_hi, a_lo = int(np.argmax(rew - (lam - 1e-6) * cost)), int(np.argmax(rew - (lam + 1e-6) * cost))
q = (budget - cost[a_lo]) / (cost[a_hi] - cost[a_lo])   # 두 선택을 섞어 예산을 정확히 맞춘다
print(f"λ* = {lam:.3f}, 섞는 조건 {a_hi}(확률 {q:.2f})와 {a_lo}(확률 {1 - q:.2f})")
print(f"혼합 정책: 기대 보상 {q * rew[a_hi] + (1 - q) * rew[a_lo]:.3f}, 기대 비용 {q * cost[a_hi] + (1 - q) * cost[a_lo]:.3f}")
print(f"결정적 정책 중 최선: 조건 1, 기대 보상 {rew[1]:.3f}")
