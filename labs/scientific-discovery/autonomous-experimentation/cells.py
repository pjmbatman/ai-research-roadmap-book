"""25.4 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/scientific-discovery/autonomous-experimentation/cells.py
"""
# %% cell 1 — 가상 실험 장비와 대리모델
import numpy as np
from scipy.stats import norm

def true_yield(u):                                    # u: (..., 2) 정규화한 (온도, 시간) -> 참 수율 0–1
    a = 0.8 * np.exp(-((u[..., 0] - 0.65) ** 2 / 0.05 + (u[..., 1] - 0.4) ** 2 / 0.08))
    b = 0.55 * np.exp(-((u[..., 0] - 0.2) ** 2 + (u[..., 1] - 0.8) ** 2) / 0.02)
    return a + b

def units(u):                                         # 정규화 좌표 -> (온도 °C, 시간 min)
    return 20 + 100 * u[0], 5 + 55 * u[1]

g = np.linspace(0, 1, 21)
cand = np.array([(a, b) for a in g for b in g])       # (441, 2) 후보 실험 조건
NOISE = 0.02                                          # 측정 잡음 표준편차

def gp(X, y, Xs, ell=0.15, amp=0.3):                  # RBF 커널 GP: 사후 평균·표준편차
    k = lambda A, B: amp**2 * np.exp(-0.5 * ((A[:, None] - B[None]) ** 2).sum(-1) / ell**2)
    L = np.linalg.cholesky(k(X, X) + NOISE**2 * np.eye(len(X)))   # (n, n)
    v = np.linalg.solve(L, k(X, Xs))                  # (n, C)
    m = y.mean()
    return m + v.T @ np.linalg.solve(L, y - m), np.sqrt(np.clip(amp**2 - (v**2).sum(0), 1e-12, None))

best_j = np.argmax(true_yield(cand))
print("참 최적 조건: 온도 {:.0f} °C, 시간 {:.1f} min, 수율 {:.3f}".format(*units(cand[best_j]), true_yield(cand[best_j])))
print("두 번째 봉우리 수율: {:.3f}".format(true_yield(np.array([0.2, 0.8]))))

# %% cell 2 — closed loop: 선택 → 측정 → 갱신, 모든 결정을 기록
def closed_loop(seed, n_exp=20, beta=2.0, n_init=3):
    rng = np.random.default_rng(seed)
    X, y, log = [], [], []
    for i in range(n_exp):
        if i < n_init:                                # 초기 실험은 무작위
            j, why = int(rng.integers(len(cand))), "초기 무작위"
        else:
            mu, sd = gp(np.array(X), np.array(y), cand)
            j = int(np.argmax(mu + beta * sd))        # UCB 획득함수
            why = f"UCB: 예측 {mu[j]:.3f} ± {sd[j]:.3f}"
        y_obs = float(true_yield(cand[j]) + NOISE * rng.normal())   # 가상 장비에서 측정
        X.append(cand[j]); y.append(y_obs)
        log.append({"실험": i, "온도_c": round(float(units(cand[j])[0]), 1), "시간_min": round(float(units(cand[j])[1]), 2),
                    "근거": why, "측정": round(y_obs, 3)})
    return log

log = closed_loop(seed=1)
print("같은 시드로 다시 돌린 결정 기록이 같다:", log == closed_loop(seed=1))
for r in log[2:5] + log[-1:]:
    print(r)
best = max(log, key=lambda r: r["측정"])
print(f"최고 측정: 실험 {best['실험']}, {best['온도_c']} °C, {best['시간_min']} min, 수율 {best['측정']}")

# %% cell 3 — batch 선택: 장비 4대를 한 번에 돌릴 때
def pick_batch(X, y, q, beta=2.0, believer=True):
    X, y = list(X), list(y)
    if not believer:                                  # 단순 상위 q개: 한 번 계산한 UCB 점수의 상위 q
        mu, sd = gp(np.array(X), np.array(y), cand)
        return [int(j) for j in np.argsort(-(mu + beta * sd))[:q]]
    batch = []
    for _ in range(q):
        mu, sd = gp(np.array(X), np.array(y), cand)
        j = int(np.argmax(mu + beta * sd))
        batch.append(j)
        X.append(cand[j]); y.append(mu[j])            # kriging believer: 예측 평균을 측정값인 척 넣는다
    return batch

def run_batch(seed, q, rounds, believer=True):
    rng = np.random.default_rng(seed)
    idx = [int(j) for j in rng.integers(len(cand), size=3)]          # 초기 3개는 무작위
    y = [float(true_yield(cand[j]) + NOISE * rng.normal()) for j in idx]
    spread = []
    for _ in range(rounds):
        b = pick_batch(cand[idx], y, q, believer=believer)
        if q > 1:                                     # batch 안에서 가장 가까운 두 조건의 거리
            spread.append(min(np.linalg.norm(cand[i] - cand[j]) for i in b for j in b if i < j))
        for j in b:
            idx.append(j); y.append(float(true_yield(cand[j]) + NOISE * rng.normal()))
    best = np.maximum.accumulate(true_yield(cand[idx]))             # 지금까지 고른 조건의 참 수율 최고
    regret = true_yield(cand).max() - best[[3 + 4 - 1, 3 + 8 - 1, 3 + 16 - 1]]   # 실험 4·8·16개 뒤
    return list(regret) + [np.mean(spread) if spread else np.nan]

for name, q, rounds, bel in [("순차 1개씩", 1, 16, True), ("4개 batch, believer", 4, 4, True),
                             ("4개 batch, 단순 상위 4", 4, 4, False)]:
    R = np.array([run_batch(s, q, rounds, bel) for s in range(20)])    # (시드, 4)
    gap = f"{np.nanmean(R[:, 3]):.3f}" if q > 1 else "  -  "
    print(f"{name:18s} 라운드 {rounds:2d}회  평균 regret  실험 4개 {R[:, 0].mean():.3f}  8개 {R[:, 1].mean():.3f}  16개 {R[:, 2].mean():.3f}  batch 최소 간격 {gap}")

# %% cell 4 — 안전 제약: 압력 한계를 넘는 실험을 피하기
P_MAX = 4.0                                           # 반응기 압력 한계 [bar]
def pressure_bar(u):                                  # 장비가 함께 기록하는 압력(실험 전에는 모른다)
    return 1 + 8 * u[..., 0] * (0.3 + u[..., 1])

feasible = pressure_bar(cand) < P_MAX
print(f"제약 없는 최고 수율 {true_yield(cand).max():.3f}, 압력 한계 안 최고 수율 {true_yield(cand[feasible]).max():.3f}")

def constrained_loop(seed, use_constraint, n_exp=20):
    rng = np.random.default_rng(seed)
    idx = [int(j) for j in rng.choice(np.flatnonzero(feasible), 3)]  # 초기 3개는 안전한 조건에서
    y = [float(true_yield(cand[j]) + NOISE * rng.normal()) for j in idx]
    p = [float(pressure_bar(cand[j])) for j in idx]
    while len(idx) < n_exp:
        mu, sd = gp(cand[idx], np.array(y), cand)
        z = (mu - max(y)) / sd
        score = (mu - max(y)) * norm.cdf(z) + sd * norm.pdf(z)       # EI (최대화)
        if use_constraint:                                           # 압력 GP로 한계 안일 확률을 곱한다
            mp, sp = gp(cand[idx], np.array(p), cand, ell=0.4, amp=2.0)
            score = score * norm.cdf((P_MAX - mp) / sp)
        j = int(np.argmax(score))
        safe = pressure_bar(cand[j]) < P_MAX
        idx.append(j); p.append(float(pressure_bar(cand[j])))
        y.append(float(true_yield(cand[j]) + NOISE * rng.normal()) if safe else 0.0)   # 중단된 실험은 수율 0으로 기록
    ok = [j for j in idx if feasible[j]]
    return int((~feasible[idx]).sum()), true_yield(cand[feasible]).max() - true_yield(cand[ok]).max()

for use in (False, True):
    R = np.array([constrained_loop(s, use) for s in range(20)])     # (시드, 2)
    print(f"제약 모델 {'사용' if use else '없음'}: 한계 위반 실험 평균 {R[:, 0].mean():.1f}/20회, 안전 영역 regret 평균 {R[:, 1].mean():.3f}")

# %% cell 5 — 측정 실패와 이상값: 상태 확인 없이 학습하면
def instrument(j, rng):                               # 가상 장비: (측정값, 상태)
    if rng.random() < 0.15:
        return np.nan, "실패"                          # 시료 손실 등으로 값이 없다
    if cand[j, 0] > 0.85 and cand[j, 1] < 0.2:        # 고온·단시간 영역에서 검출기가 포화된다
        return 1.0, "포화"
    return float(true_yield(cand[j]) + NOISE * rng.normal()), "정상"

def robust_loop(seed, check_status, n_exp=20, beta=2.0):
    rng = np.random.default_rng(seed)
    idx, y, banned = [], [], set()
    for _ in range(n_exp):                            # 실패한 실험도 예산을 쓴다
        if len(idx) < 3:                              # 쓸 수 있는 측정이 3개가 될 때까지 무작위
            j = int(rng.integers(len(cand)))
        else:
            mu, sd = gp(cand[idx], np.array(y), cand)
            score = mu + beta * sd
            score[list(banned)] = -np.inf             # 포화가 확인된 조건은 다시 고르지 않는다
            j = int(np.argmax(score))
        val, status = instrument(j, rng)
        if check_status and status != "정상":         # 상태를 확인하면: 학습에서 빼고 기록만 남긴다
            banned |= {j} if status == "포화" else set()
            continue
        idx.append(j); y.append(0.0 if np.isnan(val) else val)       # 확인하지 않으면: 실패는 0, 포화값은 그대로
    k = int(np.argmax(y))                             # 루프가 보고하는 최고 조건
    return y[k], true_yield(cand[idx[k]]), sum(cand[j, 0] > 0.85 and cand[j, 1] < 0.2 for j in idx)

for check in (False, True):
    R = np.array([robust_loop(s, check) for s in range(20)])        # (시드, 3)
    print(f"상태 확인 {'함' if check else '안 함'}: 보고한 최고 수율 평균 {R[:, 0].mean():.3f}, "
          f"그 조건의 참 수율 평균 {R[:, 1].mean():.3f}, 학습에 들어간 포화 영역 실험 평균 {R[:, 2].mean():.1f}회")
