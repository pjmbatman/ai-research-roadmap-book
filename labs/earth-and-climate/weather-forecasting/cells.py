"""27.2 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/earth-and-climate/weather-forecasting/cells.py
"""
# %% cell 1 — Lorenz-96 자료와 두 baseline
import numpy as np

K, F, DT = 40, 8.0, 0.05                     # 격자점 수, 강제력, 한 step(약 6시간에 해당)

def tend(x):                                 # x: (..., K) 위도권 한 바퀴의 격자점
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + F

def step(x):                                 # 4차 Runge–Kutta 한 step
    k1 = tend(x); k2 = tend(x + DT / 2 * k1); k3 = tend(x + DT / 2 * k2); k4 = tend(x + DT * k3)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

rng = np.random.default_rng(0)
x = F + rng.normal(0, 1, K)
for _ in range(1000):                        # spin-up: 초기값의 흔적을 지운다
    x = step(x)
traj = np.empty((12000, K))                  # (T, K) 약 8년치 6시간 간격 자료
for t in range(len(traj)):
    traj[t] = x = step(x)

train, test = traj[:8000], traj[8400:]       # 시간 순서 분할, 사이 400 step은 버린다(gap)
clim = train.mean()                          # 학습 기간의 기후값(격자점 모두 같은 통계)
starts = np.arange(0, len(test) - 41, 10)    # (N,) 시험 기간의 예보 시작 시점
truth = np.stack([test[starts + h] for h in range(41)], 1)   # (N, 41, K) 리드 0–40 step
rmse = lambda f: np.sqrt(((f - truth) ** 2).mean(axis=(0, 2)))   # 리드별 RMSE (41,)
r_pers, r_clim = rmse(truth[:, :1]), rmse(np.full_like(truth, clim))
for h in [1, 4, 8, 20, 40]:
    print(f"리드 {h:2d} step  지속성 {r_pers[h]:.2f}  기후값 {r_clim[h]:.2f}")

# %% cell 2 — 한 step 사상을 학습하고 굴리기
def feats(x):                                # x: (..., K) → (..., K, 15) 국소 stencil의 1·2차 항
    s = np.stack([np.roll(x, k, -1) for k in (2, 1, 0, -1)], -1)   # (..., K, 4) x_{i-2..i+1}
    quad = [s[..., a] * s[..., b] for a in range(4) for b in range(a, 4)]
    return np.concatenate([np.ones(s.shape[:-1] + (1,)), s, np.stack(quad, -1)], -1)

def fit(noise_sd):                           # 입력에 잡음을 넣고 다음 상태를 맞히도록 학습
    x_in = train[:-1] + np.random.default_rng(1).normal(0, noise_sd, train[:-1].shape)
    coef = np.linalg.lstsq(feats(x_in).reshape(-1, 15), (train[1:] - x_in).ravel(), rcond=None)[0]
    return lambda x: x + feats(x) @ coef     # 모든 격자점이 계수 15개를 공유한다

def rollout(model, x0, n=40):                # 자기 출력을 다음 입력으로 넣는다
    out = [x0]
    with np.errstate(all="ignore"):
        for _ in range(n):
            out.append(model(out[-1]))
    return np.stack(out, 1)                  # (N, n+1, K)

for sd in [0.0, 0.7, 1.5]:
    fc = rollout(fit(sd), truth[:, 0])
    with np.errstate(all="ignore"):
        r = rmse(fc)
    bad = (~np.isfinite(fc[:, -1]).all(-1)).sum()
    print(f"입력 잡음 {sd}: RMSE 리드1 {r[1]:.2f} 리드8 {r[8]:.2f} 리드20 {r[20]:.2f}  발산 {bad}/{len(fc)}")

# %% cell 3 — 리드 타임별 RMSE·ACC·스킬 점수
fc = rollout(fit(0.7), truth[:, 0])                  # (N, 41, K) 안정적으로 굴러간 모델
r_model = rmse(fc)
an_f, an_o = fc - clim, truth - clim                 # 학습 기간 기후값에 대한 편차
acc = ((an_f * an_o).sum(-1) / np.sqrt((an_f**2).sum(-1) * (an_o**2).sum(-1))).mean(0)  # (41,)
print("리드  모델RMSE  ACC   스킬(지속성)  스킬(기후값)")
for h in [1, 4, 8, 12, 20, 40]:
    ss_p = 1 - r_model[h]**2 / r_pers[h]**2          # MSE 기반 스킬 점수
    ss_c = 1 - r_model[h]**2 / r_clim[h]**2
    print(f"{h:3d}   {r_model[h]:6.2f}  {acc[h]:5.2f}   {ss_p:+8.2f}   {ss_c:+8.2f}")

# %% cell 4 — 완벽한 모델이라도: 초기조건 앙상블
def roll_true(x0, n=40):                             # 참 동역학으로 굴린다(모델 오차 없음)
    out = [x0]
    for _ in range(n):
        out.append(step(out[-1]))
    return np.stack(out, -2)                         # (..., n+1, K)

g = np.random.default_rng(2)
analysis = truth[:, 0] + g.normal(0, 0.5, truth[:, 0].shape)          # (N, K) 분석 오차 0.5
members = roll_true(analysis[None] + g.normal(0, 0.5, (20,) + analysis.shape))  # (20, N, 41, K)
control = roll_true(analysis)                                         # (N, 41, K)
mean = members.mean(0)
spread = np.sqrt(members.var(0, ddof=1).mean(axis=(0, 2)))            # (41,)
M, xs = len(members), np.sort(members, 0)                           # 정렬하면 E|X-X'|를 O(M)으로 계산
pair = 2 * ((2 * np.arange(1, M + 1) - M - 1)[:, None, None, None] * xs).sum(0) / M**2
crps = (np.abs(members - truth).mean(0) - pair / 2).mean(axis=(0, 2))  # (41,) 표본 CRPS
mae_c = np.abs(control - truth).mean(axis=(0, 2))
print("리드  단일RMSE  평균RMSE  spread  CRPS  단일MAE")
for h in [1, 8, 20, 40]:
    print(f"{h:3d}   {rmse(control)[h]:6.2f}   {rmse(mean)[h]:6.2f}  {spread[h]:6.2f} {crps[h]:5.2f}  {mae_c[h]:5.2f}")
