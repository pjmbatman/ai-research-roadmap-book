"""30.6 실습: 3단계 충전 프로토콜을 시뮬레이터에서 최적화하고, 파라미터가 다른 "실제 셀"에서 제약 위반을 평가한다.

시뮬레이터: 본문 cell 2와 같은 1-RC 등가회로 + 집중 열 모델 + 가상 열화 지표(설명용, 측정한 열화가 아니다).
"실제 셀": 같은 모델에서 저항·열전달 계수·주변 온도를 바꾼 가상의 셀 200개(셀 편차·노화·환경을 흉내 낸다).
비교: (1) 공칭 셀·공칭 제약(4.2 V, 45 °C)으로 고른 최적 프로토콜, (2) 공칭 셀에 여유를 둔 제약(4.15 V, 40 °C),
(3) 최악 조건 셀(저항 1.4배, 열전달 0.7배, 주변 35 °C)에서 제약을 확인한 프로토콜(충전 시간 한계 20 min, 35 min).
두 프로토콜 모두 3단계 C-rate 후보 13^3 = 2197개 전수 탐색과, 같은 후보에서 평가 30번의 제약 BO로 찾는다.

실행: uv run --project labs python labs/battery-ai/charging-optimization/sim_to_real.py
"""
import numpy as np
from scipy.stats import norm

Q_AH, C1_F, V_MAX_V, MCP_JPK = 3.0, 2000.0, 4.2, 45.0
EDGES = (0.1, 0.4, 0.6, 0.8)                              # SOC 구간 경계: 단계 1, 2, 3


def ocv_v(soc):
    return 3.5 + 0.55 * soc + 0.04 * np.log((soc + 0.02) / (1.02 - soc))


def simulate(rates, r0_ohm=0.020, r1_ohm=0.015, ha_wpk=0.10, t_amb_c=25.0, dt_s=2.0):
    """rates: (N, 3) 단계별 C-rate. 반환: 시간(min), 열화 지표, 최고 전압(V), 최고 온도(°C), 각 (N,)"""
    n = len(rates)
    soc, v1, temp = np.full(n, EDGES[0]), np.zeros(n), np.full(n, t_amb_c)
    t, wear, v_pk, t_pk = np.zeros(n), np.zeros(n), np.zeros(n), temp.copy()
    a = np.exp(-dt_s / (r1_ohm * C1_F))
    while np.any(soc < EDGES[-1]):
        stage = np.searchsorted(EDGES[1:-1], soc, side="right")       # (N,) 0, 1, 2
        i = np.where(soc < EDGES[-1], rates[np.arange(n), np.minimum(stage, 2)] * Q_AH, 0.0)
        v_pk = np.maximum(v_pk, np.where(i > 0, ocv_v(soc) + i * r0_ohm + v1, 0.0))
        wear += (i / Q_AH) ** 2 * np.exp(3 * (soc - 0.5) + (temp - 25.0) / 15) * dt_s / 3600
        temp += dt_s * (i * (i * r0_ohm + v1) - ha_wpk * (temp - t_amb_c)) / MCP_JPK
        v1 = a * v1 + r1_ohm * (1 - a) * i
        t += np.where(i > 0, dt_s, 0.0)
        soc += i * dt_s / (3600 * Q_AH)
        t_pk = np.maximum(t_pk, temp)
    return t / 60, wear, v_pk, t_pk


def gp(X, y, Xs, ell=0.15, noise=1e-6):
    k = lambda A, B: np.exp(-0.5 * np.sum((A[:, None] - B[None]) ** 2, -1) / ell**2)
    m, s = y.mean(), y.std() + 1e-12
    L = np.linalg.cholesky(k(X, X) + noise * np.eye(len(X)))
    v = np.linalg.solve(L, k(X, Xs))
    return m + s * (v.T @ np.linalg.solve(L, (y - m) / s)), s * np.sqrt(np.clip(1 - np.sum(v**2, 0), 1e-12, None))


def constrained_bo(X, f, g, budget, seed):
    """후보 X (C, d)에서 목적 f를 g ≤ 0 아래 최소화. 평가한 점에서만 f, g를 본다."""
    rng = np.random.default_rng(seed)
    idx = list(rng.choice(len(X), 5, replace=False))
    while len(idx) < budget:
        I = np.array(idx)
        mu_g, sd_g = gp(X[I], g[I], X)
        pof = norm.cdf(-mu_g / sd_g)
        ok = g[I] <= 0
        if ok.any():
            mu, sd = gp(X[I], f[I], X)
            z = (f[I][ok].min() - mu) / sd
            score = sd * (z * norm.cdf(z) + norm.pdf(z)) * pof
        else:
            score = pof
        score[I] = -np.inf
        idx.append(int(np.argmax(score)))
    I = np.array(idx)
    ok = g[I] <= 0
    return int(I[ok][np.argmin(f[I][ok])]) if ok.any() else None


def main():
    levels = np.arange(1.0, 4.01, 0.25)                   # 13개 C-rate 수준
    rates = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), -1).reshape(-1, 3)   # (2197, 3)
    t_min, wear, v_pk, t_pk = simulate(rates)
    X = (rates - 1.0) / 3.0
    f = np.log(wear)

    rng = np.random.default_rng(1)
    n_cell = 200
    cells = dict(r0_ohm=0.020 * rng.uniform(0.9, 1.4, n_cell), r1_ohm=0.015 * rng.uniform(0.9, 1.4, n_cell),
                 ha_wpk=0.10 * rng.uniform(0.7, 1.1, n_cell), t_amb_c=rng.uniform(20, 35, n_cell))

    print(f"후보 {len(rates)}개, 시뮬레이터에서 4.2 V·45 °C·20 min을 모두 만족하는 것 "
          f"{np.sum((v_pk <= 4.2) & (t_pk <= 45) & (t_min <= 20))}개")
    _, _, v_pk_w, t_pk_w = simulate(rates, r0_ohm=0.028, r1_ohm=0.021, ha_wpk=0.07, t_amb_c=35.0)   # 최악 모서리
    strategies = [("공칭 셀, 공칭 제약 4.2 V·45 °C, 20 min", v_pk, t_pk, 4.2, 45.0, 20.0),
                  ("공칭 셀, 여유 제약 4.15 V·40 °C, 20 min", v_pk, t_pk, 4.15, 40.0, 20.0),
                  ("최악 조건 셀(R ×1.4, h ×0.7, 35 °C), 4.2 V·45 °C, 20 min", v_pk_w, t_pk_w, 4.2, 45.0, 20.0),
                  ("최악 조건 셀(R ×1.4, h ×0.7, 35 °C), 4.2 V·45 °C, 35 min", v_pk_w, t_pk_w, 4.2, 45.0, 35.0)]
    for name, vp, tp, v_lim, temp_lim, time_lim in strategies:
        g = np.maximum.reduce([(vp - v_lim) / 0.05, (tp - temp_lim) / 5.0, (t_min - time_lim) / 2.0])
        if not np.any(g <= 0):
            print(f"\n{name}\n  제약을 모두 만족하는 프로토콜이 없다")
            continue
        k = int(np.flatnonzero(g <= 0)[np.argmin(f[g <= 0])])
        found = [constrained_bo(X, f, g, 30, s) for s in range(10)]
        hit = np.mean([b is not None and f[b] - f[k] < np.log(1.05) for b in found])
        # 같은 프로토콜을 200개 "실제 셀"에 적용
        real = [simulate(rates[k:k + 1], **{p: v[c] for p, v in cells.items()}) for c in range(n_cell)]
        t_r, w_r, v_r, T_r = (np.array([r[j][0] for r in real]) for j in range(4))
        print(f"\n{name}")
        print(f"  시뮬레이터 최적: {rates[k][0]:.2f}C → {rates[k][1]:.2f}C → {rates[k][2]:.2f}C, "
              f"시간 {t_min[k]:.1f} min, 열화 {wear[k]:.3f}, 최고 {v_pk[k]:.3f} V, {t_pk[k]:.1f} °C")
        print(f"  제약 BO(평가 30번, 시드 10개)가 최적의 5% 이내를 찾은 비율 {hit:.1f}")
        print(f"  실제 셀 200개: 4.2 V 초과 {np.mean(v_r > 4.2):.0%}, 45 °C 초과 {np.mean(T_r > 45):.0%}, "
              f"{time_lim:.0f} min 초과 {np.mean(t_r > time_lim):.0%}, 열화 지표 중앙값 {np.median(w_r):.3f} "
              f"(5–95% {np.percentile(w_r, 5):.3f}–{np.percentile(w_r, 95):.3f})")


if __name__ == "__main__":
    main()
