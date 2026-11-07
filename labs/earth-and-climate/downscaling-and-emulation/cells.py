"""27.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/earth-and-climate/downscaling-and-emulation/cells.py
"""
# %% cell 1 — 산악 단면의 저해상도–고해상도 쌍과 회귀 다운스케일링
import numpy as np

rng = np.random.default_rng(0)
n_day, n_fine, blk = 4000, 64, 8                      # 일 수, 고해상도 칸 수, 저해상도 칸 하나에 든 고해상도 칸 수
x_km = np.arange(n_fine) * 2.0                        # (64,) 2 km 간격 단면
elev_m = 200 + 1400 * np.abs(np.sin(x_km / 9)) ** 3 + 600 * (x_km > 64)   # (64,) 봉우리와 골짜기, 동쪽 고원
large = np.zeros(n_day)                               # 대규모 기온 편차: 하루 간격 AR(1)
for t in range(1, n_day):
    large[t] = 0.8 * large[t - 1] + rng.normal(0, 2.0)
eps = rng.normal(0, 1, (n_day, n_fine))
local = np.empty_like(eps); local[:, 0] = eps[:, 0]
for j in range(1, n_fine):                            # 국지 변동: 이웃 칸끼리 상관(0.8)된 잡음
    local[:, j] = 0.8 * local[:, j - 1] + 0.6 * eps[:, j]
fine_c = 15 - 0.0065 * elev_m + large[:, None] * (1 + elev_m / 3000) + 1.5 * local   # (n_day, 64)
coarse_c = fine_c.reshape(n_day, -1, blk).mean(-1)   # (n_day, 8) 저해상도 = 칸 평균

tr, te = slice(0, 3000), slice(3000, None)            # 앞 3000일로 학습, 나머지 1000일로 시험
X = np.repeat(coarse_c, blk, axis=1)                  # (n_day, 64) 각 고해상도 칸이 속한 저해상도 값
coef = [np.polyfit(X[tr, j], fine_c[tr, j], 1) for j in range(n_fine)]      # 칸마다 1차 회귀
pred = np.stack([np.polyval(coef[j], X[te, j]) for j in range(n_fine)], 1)  # (1000, 64)
rmse = lambda a: np.sqrt(((a - fine_c[te]) ** 2).mean())
print(f"저해상도 값 그대로  RMSE {rmse(X[te]):.2f} °C")
print(f"칸별 회귀           RMSE {rmse(pred):.2f} °C")

# %% cell 2 — quantile mapping 편향 보정과 그 한계
def daily_precip(g, n, p_wet, shape, scale_mm):       # 강수일 확률 × 감마 분포 강수량
    return np.where(g.random(n) < p_wet, g.gamma(shape, scale_mm, n), 0.0)

g = np.random.default_rng(2)
n = 30 * 365                                          # 30년치 일강수 [mm/day]
obs_h = daily_precip(g, n, 0.30, 0.7, 9.0)            # 관측(과거)
mod_h = daily_precip(g, n, 0.50, 1.0, 3.5)            # 모델(과거): 약한 비가 너무 잦고 강한 비가 약하다
mod_f = daily_precip(g, n, 0.50, 1.0, 3.5 * 1.3)      # 모델(미래): 강수 강도 30% 증가

qs = np.linspace(0, 1, 1001)
mq, oq = np.quantile(mod_h, qs), np.quantile(obs_h, qs)    # (1001,) 과거 기간의 분위수
qm = lambda x: np.interp(x, mq, oq)                   # 모델 값 → 같은 분위의 관측 값(범위 밖은 끝값 고정)

stat = lambda x: f"강수일 {np.mean(x > 0.1):.2f}  평균 {x.mean():5.2f}  99%분위 {np.percentile(x, 99):6.2f}"
print("과거 관측   ", stat(obs_h))
print("과거 모델   ", stat(mod_h))
print("과거 모델+QM", stat(qm(mod_h)))
chg = lambda stat_fn, f, h: 100 * (stat_fn(f) / stat_fn(h) - 1)     # 미래/과거 변화율 [%]
for label, s in [("평균", np.mean), ("99.9%분위", lambda x: np.percentile(x, 99.9))]:
    print(f"미래 변화 {label}: 모델 {chg(s, mod_f, mod_h):+.0f}%  QM 후 {chg(s, qm(mod_f), qm(mod_h)):+.0f}%")
print(f"미래 값 중 과거 모델 최댓값을 넘는 비율 {np.mean(mod_f > mq[-1]):.4f}")

# %% cell 3 — 평균 오차 밖의 평가: 분산·극한·공간 구조
res = fine_c[tr] - np.stack([np.polyval(coef[j], X[tr, j]) for j in range(n_fine)], 1)  # (3000, 64) 학습 잔차
sd = res.std(0)                                       # (64,) 칸별 잔차 표준편차
phi = np.corrcoef(res[:, :-1].ravel(), res[:, 1:].ravel())[0, 1]    # 잔차의 이웃 칸 상관
z = np.random.default_rng(1).normal(0, 1, pred.shape)
cor = np.empty_like(z); cor[:, 0] = z[:, 0]
for j in range(1, n_fine):                            # 이웃 상관 phi를 가진 잡음
    cor[:, j] = phi * cor[:, j - 1] + np.sqrt(1 - phi**2) * z[:, j]

obs = fine_c[te]                                      # (1000, 64)
def detail(f):                                        # 저해상도 값을 뺀 국지 편차에서 날마다 달라지는 부분
    d = f - X[te]
    return d - d.mean(0)

print(f"학습 잔차의 이웃 칸 상관 phi = {phi:.2f}")
print(" RMSE  국지변동 표준편차비  99%분위 편향  국지변동 이웃상관  방법")
for name, f in [("정답", obs), ("회귀", pred), ("회귀+독립잡음", pred + sd * z), ("회귀+상관잡음", pred + sd * cor)]:
    d, q99 = detail(f), (np.percentile(f, 99, axis=0) - np.percentile(obs, 99, axis=0)).mean()
    lag1 = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    print(f"{rmse(f):5.2f}   {(d.std(0) / detail(obs).std(0)).mean():10.2f}"
          f"   {q99:+11.2f}   {lag1:14.2f}     {name}")

# %% cell 4 — 에뮬레이터의 외삽: 강제력 → 전구 평균 기온 변화
true_resp = lambda f: 0.75 * f + 0.04 * f**2          # '기후 모델'의 반응 [K], 되먹임이 강제력에 따라 커진다
g = np.random.default_rng(3)
f_tr = g.uniform(0, 4.5, 200)                         # (200,) 학습 시나리오의 강제력 [W/m^2]
t_tr = true_resp(f_tr) + g.normal(0, 0.15, 200)       # 내부 변동성이 섞인 모의 결과 [K]

def knn(f, k=10):                                     # 최근접 시나리오 k개의 평균
    idx = np.argsort(np.abs(f_tr[None] - np.atleast_1d(f)[:, None]), 1)[:, :k]
    return t_tr[idx].mean(1)

emus = {"선형": np.poly1d(np.polyfit(f_tr, t_tr, 1)),
        "2차": np.poly1d(np.polyfit(f_tr, t_tr, 2)),
        "k-NN": knn}
f_val = g.uniform(0, 4.5, 100)                        # 학습 범위 안의 검증 시나리오
t_val = true_resp(f_val) + g.normal(0, 0.15, 100)
f_te = np.array([2.0, 4.0, 6.0, 8.5])                 # 4.5를 넘는 값은 학습 범위 밖
print("범위 안 검증 RMSE |  강제력 " + "  ".join(f"{f:5.1f}" for f in f_te) + "  W/m^2")
print("                  |  참 반응 " + "  ".join(f"{v:5.2f}" for v in true_resp(f_te)) + "  K")
for name, emu in emus.items():
    r = np.sqrt(np.mean((emu(f_val) - t_val) ** 2))
    print(f"{r:10.3f}        |  {name:6s}" + "  ".join(f"{v:5.2f}" for v in emu(f_te)))
