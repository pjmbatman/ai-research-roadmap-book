"""20.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/reinforcement-learning/model-free-rl/cells.py
"""
# %% cell 1 — 무작위 보행에서 TD(0)와 Monte Carlo 정책 평가
import numpy as np

V_TRUE = np.arange(1, 6) / 6                 # 상태 A..E의 참 가치 1/6..5/6 (오른쪽 끝 보상 1)

def episode(rng):                            # 가운데(C)에서 출발해 좌우로 1/2씩 걷는다
    s, path = 2, []
    while 0 <= s <= 4:
        path.append(s)
        s += 1 if rng.random() < 0.5 else -1
    return path, float(s == 5)               # 방문한 상태들, 종료 보상

def rms_after(method, alpha, n_ep=100, n_run=50):
    err = 0.0
    for run in range(n_run):
        rng, V = np.random.default_rng(run), np.full(5, 0.5)
        for _ in range(n_ep):
            path, G = episode(rng)
            for t, s in enumerate(path):
                if method == "MC":           # 목표: 실제 이득 G (할인 없음)
                    target = G
                else:                        # TD(0) 목표: r + V(s'), 종료 시 r
                    target = V[path[t + 1]] if t + 1 < len(path) else G
                V[s] += alpha * (target - V[s])
        err += np.sqrt(np.mean((V - V_TRUE) ** 2)) / n_run
    return err

for method, alpha in [("MC", 0.02), ("MC", 0.05), ("TD", 0.05), ("TD", 0.1)]:
    print(f"{method}  alpha={alpha:.2f}  100 에피소드 뒤 RMS 오차 {rms_after(method, alpha):.3f}")

# %% cell 2 — 절벽 걷기: Q-learning과 SARSA
H, W, START, GOAL = 4, 12, 36, 47            # 4×12 격자, 아래 줄 가운데 10칸이 절벽
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
def step(s, a):
    r, c = divmod(s, W)
    r, c = min(max(r + MOVES[a][0], 0), H - 1), min(max(c + MOVES[a][1], 0), W - 1)
    if r == H - 1 and 0 < c < W - 1:         # 절벽: 보상 -100, 출발점으로
        return START, -100.0
    return r * W + c, -1.0
def learn(method, seed, n_ep=300, alpha=0.5, eps=0.1):
    rng, Q, returns = np.random.default_rng(seed), np.zeros((H * W, 4)), []
    pick = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(n_ep):
        s, a, G = START, pick(START), 0.0
        while s != GOAL:
            s2, r = step(s, a)
            a2 = pick(s2)
            target = r + (0.0 if s2 == GOAL else
                          (Q[s2].max() if method == "Q" else Q[s2, a2]))   # off- / on-policy 목표
            Q[s, a] += alpha * (target - Q[s, a])
            s, a, G = s2, a2, G + r
        returns.append(G)
    return Q, np.array(returns)
for method in ["Q", "SARSA"]:
    res = [learn(method, seed) for seed in range(5)]
    online = np.mean([ret[-100:].mean() for _, ret in res])
    s, n = START, 0                          # 학습된 Q의 탐욕 경로 길이(첫 seed)
    while s != GOAL and n < 100:
        s, _ = step(s, int(np.argmax(res[0][0][s])))
        n += 1
    print(f"{method:5s}  학습 중 이득(마지막 100 에피소드, 5 seed 평균) {online:6.1f}   탐욕 경로 {n} step")

# %% cell 3 — REINFORCE 기울기 추정량의 분산과 baseline
logits = np.array([0.0, 0.0, 0.0])           # softmax 정책의 파라미터, 행동 3개
pi = np.exp(logits) / np.exp(logits).sum()   # (3,) 각 행동 확률 1/3
mu = np.array([0.5, 1.2, 0.9])               # 행동별 평균 보상
grad_true = pi * (mu - pi @ mu)              # 기대 보상의 정확한 기울기, (3,)

rng = np.random.default_rng(0)
N = 200_000
a = rng.choice(3, size=N, p=pi)              # (N,) 표집한 행동
for offset in [0.0, 10.0]:                   # 모든 보상에 같은 상수를 더해 본다
    r = mu[a] + offset + rng.normal(0, 0.5, N)          # (N,) 잡음 있는 보상
    score = np.eye(3)[a] - pi                # grad log pi(a) = onehot(a) - pi, (N, 3)
    for name, b in [("baseline 없음", 0.0), ("baseline = 기대 보상", pi @ mu + offset)]:
        g = (r - b)[:, None] * score         # 표본별 기울기 추정, (N, 3)
        err = np.abs(g.mean(axis=0) - grad_true).max()
        print(f"보상 오프셋 {offset:4.1f}  {name:16s}  평균 오차 {err:.4f}  표본 분산 합 {g.var(axis=0).sum():8.3f}")
print("정확한 기울기", grad_true.round(4))

# %% cell 4 — seed 민감도: 같은 알고리즘, 같은 설정, 다른 seed
N_S = 5                                      # 사슬 상태 0..4, 0에서 출발해 4에 닿으면 보상 1
def chain_run(seed, eps, n_ep=100, alpha=0.2, gamma=0.95):
    rng, Q = np.random.default_rng(seed), np.zeros((N_S, 2))   # 행동 0 = 전진, 1 = 포기
    for _ in range(n_ep):
        s = 0
        while True:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
            if a == 1:                       # 포기: 작은 보상 0.1을 받고 에피소드 종료
                Q[s, 1] += alpha * (0.1 - Q[s, 1])
                break
            done = s + 1 == N_S - 1
            target = 1.0 if done else gamma * Q[s + 1].max()
            Q[s, 0] += alpha * (target - Q[s, 0])
            if done:
                break
            s += 1
    return float(np.all(np.argmax(Q[:N_S - 1], axis=1) == 0))    # 탐욕 정책이 끝까지 가면 성공

res = {eps: np.array([chain_run(seed, eps) for seed in range(30)]) for eps in (0.1, 0.2)}
for eps, x in res.items():
    se = x.std(ddof=1) / np.sqrt(x.size)
    print(f"eps={eps}: seed별 성공 {''.join(str(int(v)) for v in x)}  평균 {x.mean():.2f} ± {1.96 * se:.2f}")
a, b = res[0.1], res[0.2]
print(f"seed 0 하나로 비교: eps=0.1 {a[0]:.0f}, eps=0.2 {b[0]:.0f}")
d_se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
print(f"30 seed 차이: {a.mean() - b.mean():.2f}, 95% 구간 [{a.mean() - b.mean() - 1.96 * d_se:.2f}, {a.mean() - b.mean() + 1.96 * d_se:.2f}]")
