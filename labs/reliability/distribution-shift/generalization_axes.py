"""21.4 실습: 새로운 셀·온도·충방전 조건·측정 장비 각각에 대한 일반화를 따로 평가한다.

실행: uv run --project labs python labs/reliability/distribution-shift/generalization_axes.py

합성 데이터(배터리 열화를 흉내 낸 것으로, 실제 배터리 모델이 아니다):
  셀 72개 = 온도 {15, 25, 35, 45} °C × C-rate {1, 2, 3} × 측정 장비 {A, B, C} × 반복 2
  셀마다 측정 창 25개. 특징: 온도, C-rate, 장비가 잰 내부저항 증가(장비 C는 +0.3 치우침), 전압 이완 기울기
  목표: 사이클당 용량 감소율(임의 단위). 온도에 대해 지수적으로, C-rate와 곱으로 커진다.
  셀마다 제조 편차 u_cell이 있어 같은 셀의 창들은 서로 닮았다.
모델: gradient boosting(트리 기반이라 학습 범위 밖 온도로는 외삽하지 못한다).
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

rng = np.random.default_rng(0)
temps, crates, devices = [15, 25, 35, 45], [1, 2, 3], ["A", "B", "C"]
rows = []
cell = 0
for T in temps:
    for c in crates:
        for d in devices:
            for _ in range(2):
                u = rng.normal(0, 0.25)                         # 셀 제조 편차
                for _ in range(25):
                    fade = np.exp(0.04 * (T - 25)) * (0.5 + 0.3 * c) * (1 + u) + rng.normal(0, 0.05)
                    r_meas = 0.8 * fade + rng.normal(0, 0.1) + (0.3 if d == "C" else 0.0)
                    relax = 0.5 * u + 0.2 * c + rng.normal(0, 0.1)
                    rows.append((cell, T, c, devices.index(d), r_meas, relax, fade))
                cell += 1
D = np.array(rows)
cell_id, temp_c, crate, dev = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
X, y = D[:, [1, 2, 4, 5]], D[:, 6]                              # 장비 이름은 특징에 넣지 않는다


def rmse_on(train, test):
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    m.fit(X[train], y[train])
    return np.sqrt(np.mean((m.predict(X[test]) - y[test]) ** 2))


def leave_one_out(values):
    return {v: rmse_on(values != v, values == v) for v in np.unique(values)}


print(f"샘플 {len(y)}개, 셀 {int(cell_id.max()) + 1}개.  목표의 표준편차 {y.std():.3f}\n")
rand = [rmse_on(tr, te) for tr, te in KFold(5, shuffle=True, random_state=0).split(X)]
grp = [rmse_on(tr, te) for tr, te in GroupKFold(5).split(X, groups=cell_id)]
print(f"{'일반화 대상':22s} {'평균 RMSE':>9s}  {'최악':>6s}   그룹별")
print(f"{'(무작위 창 분할)':22s} {np.mean(rand):9.3f}  {np.max(rand):6.3f}")
print(f"{'새 셀 (셀 단위 5-fold)':22s} {np.mean(grp):9.3f}  {np.max(grp):6.3f}")
for name, values, fmt in (("새 온도 (하나씩 빼기)", temp_c, "{:.0f}°C"),
                          ("새 C-rate (하나씩 빼기)", crate, "{:.0f}C"),
                          ("새 장비 (하나씩 빼기)", dev, None)):
    res = leave_one_out(values)
    labels = [(fmt.format(k) if fmt else devices[int(k)]) for k in res]
    detail = "  ".join(f"{lab} {v:.3f}" for lab, v in zip(labels, res.values()))
    print(f"{name:22s} {np.mean(list(res.values())):9.3f}  {max(res.values()):6.3f}   {detail}")
