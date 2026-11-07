"""26.5 실습: 학습된 시뮬레이터를 다섯 축(정확도·장기 안정성·물리 일관성·일반화·비용)으로 평가한다.

실행: uv run --project labs python labs/physics-and-engineering/physics-benchmarks/benchmark_table.py
- 본문과 같은 Burgers 설정, 같은 두 학습 시뮬레이터(A 직접 갱신, B 보존형 유속)
- 수치 baseline: 기준 solver를 절반 해상도(N = 32)에서 같은 시간 간격으로 돌린 것
- 축: (1) 정확도: 한 step 오차, t = 1의 장 오차, 적분량(운동에너지) 오차
       (2) 장기 안정성: t = 1, 4까지 발산 비율
       (3) 물리 일관성: 질량 변화, 최대 원리(초기 범위를 넘는 값)
       (4) 일반화: nu = 0.015 / 0.3, 진폭 2배, 해상도 N = 128에서 발산 비율과 오차
       (5) 비용: 파라미터 수, 모델 한 step(Δt = 0.02) 계산 시간
"""
import time

import numpy as np

L, DT = 2 * np.pi, 0.02


def ic(seed, N, amp=1.0):
    rng, x = np.random.default_rng(seed), np.arange(N) * L / N
    return rng.uniform(-0.3, 0.3) + amp * sum(rng.normal() / k * np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3))


def step_ref(u, nu, sub=4):
    dx = L / len(u)
    for _ in range(sub):
        ur = np.roll(u, -1)
        F = 0.25 * (u**2 + ur**2) - 0.5 * np.maximum(abs(u), abs(ur)) * (ur - u) - nu * (ur - u) / dx
        u = u - DT / sub / dx * (F - np.roll(F, 1))
    return u


def traj(seed, nu, N=64, amp=1.0, T=50, sub=4):
    out = [ic(seed, N, amp)]
    for _ in range(T):
        out.append(step_ref(out[-1], nu, sub))
    return np.array(out)


def feat_A(u, nu):
    S = np.stack([np.roll(u, k, -1) for k in (2, 1, 0, -1, -2)], -1)
    quad = [S[..., i] * S[..., j] for i in range(5) for j in range(i, 5)]
    lap = np.roll(u, -1, -1) - 2 * u + np.roll(u, 1, -1)
    return np.concatenate([S, np.stack(quad, -1), (nu * lap)[..., None]], -1)


def flux_B(u, nu):
    ur = np.roll(u, -1, -1)
    return np.stack([(u**2 + u * ur + ur**2) / 6, -nu * (ur - u), -abs(u + ur) * (ur - u)], -1)


def fit():
    tr = {(s, nu): traj(s, nu) for s in range(12) for nu in (0.03, 0.05, 0.1)}
    X = np.concatenate([t[:-1] for t in tr.values()])
    Y = np.concatenate([t[1:] for t in tr.values()])
    NU = np.repeat([k[1] for k in tr], 50)[:, None]
    w_A = np.linalg.lstsq(feat_A(X, NU).reshape(-1, 21), (Y - X).ravel(), rcond=None)[0]
    dF = flux_B(X, NU) - np.roll(flux_B(X, NU), 1, -2)
    w_B = np.linalg.lstsq(dF.reshape(-1, 3), (X - Y).ravel(), rcond=None)[0]

    def step_B(u, nu):
        F = flux_B(u, nu) @ w_B
        return u - (F - np.roll(F, 1))

    def coarse(u, nu):                               # 절반 해상도 solver: 64칸 입력을 평균해 32칸에서 풀고 다시 늘린다
        uc = u.reshape(-1, 2).mean(1)
        return np.repeat(step_ref(uc, nu, sub=2), 2)

    return {"A 직접 갱신": (lambda u, nu: u + feat_A(u, nu) @ w_A, 21),
            "B 보존형 유속": (step_B, 3),
            "solver N=32": (coarse, 0)}


def rollout(f, u0, nu, T):
    u, out = u0, [u0]
    for _ in range(T):
        u = f(u, nu)
        if not np.all(np.isfinite(u)) or abs(u).max() > 1e3:
            return None
        out.append(u)
    return np.array(out)


def main():
    models = fit()
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    seeds = range(100, 110)
    print("[1·2·3] 학습 범위 안 nu = 0.07, 시험 궤적 10개")
    print(f"{'모델':14s} {'한 step':>8s} {'장 오차 t=1':>11s} {'에너지 오차':>10s} {'발산 t=1':>8s} {'발산 t=4':>8s} {'질량 변화':>9s} {'범위 초과':>8s} {'파라미터':>8s} {'시간/step':>10s}")
    for name, (f, npar) in models.items():
        e1, eT, eE, mass, over, b1, b4 = [], [], [], [], 0, 0, 0
        for s in seeds:
            ref = traj(s, 0.07, T=200)
            e1.append(np.mean([rel(f(ref[i], 0.07), ref[i + 1]) for i in range(50)]))
            out = rollout(f, ref[0], 0.07, 200)
            if out is None or not np.all(np.isfinite(out[:51])):
                b4 += 1
                short = rollout(f, ref[0], 0.07, 50)
                if short is None:
                    b1 += 1
                    continue
                out = short
            eT.append(rel(out[50], ref[50]))
            eE.append(abs(np.mean(out[50] ** 2) / np.mean(ref[50] ** 2) - 1))
            mass.append(abs(out[50].mean() - ref[0].mean()) * L)
            over += int(out[50].max() > ref[0].max() + 1e-9 or out[50].min() < ref[0].min() - 1e-9)
        u0 = ic(0, 64)
        t0 = time.perf_counter()
        for _ in range(2000):
            f(u0, 0.07)
        t_step = (time.perf_counter() - t0) / 2000
        mtxt = "< 1e-12" if max(mass, default=0) < 1e-12 else f"{max(mass):.1e}"
        print(f"{name:14s} {np.mean(e1):8.4f} {np.mean(eT):11.3f} {np.mean(eE):10.3f} {b1:>5d}/10 {b4:>5d}/10 {mtxt:>9s} {over:>5d}/{len(eT)} {npar:>8d} {t_step * 1e6:7.1f} µs")

    u0 = ic(0, 64)
    t0 = time.perf_counter()
    for _ in range(2000):
        step_ref(u0, 0.07)
    print(f"참고: 기준 solver(N = 64, 하위 step 4회) 한 step {(time.perf_counter() - t0) / 2000 * 1e6:.1f} µs")
    print("[4] 일반화: 발산 비율, (발산하지 않은 궤적의) t = 1 장 오차")
    cases = [("nu = 0.015", 0.015, 64, 1.0), ("nu = 0.3", 0.3, 64, 1.0), ("진폭 2배", 0.07, 64, 2.0), ("해상도 N=128", 0.07, 128, 1.0)]
    print(f"{'모델':14s} " + " ".join(f"{c[0]:>18s}" for c in cases))
    for name, (f, _) in models.items():
        cells = []
        for _, nu, N, amp in cases:
            if N != 64 and name == "solver N=32":
                f_case = lambda u, nu: np.repeat(step_ref(u.reshape(-1, 4).mean(1), nu, sub=2), 4)
            else:
                f_case = f
            errs, blown = [], 0
            for s in seeds:
                ref = traj(s, nu, N=N, amp=amp)
                out = rollout(f_case, ref[0], nu, 50)
                if out is None:
                    blown += 1
                else:
                    errs.append(rel(out[-1], ref[-1]))
            cells.append(f"{blown:2d}/10 " + (f"{np.mean(errs):9.3f}" if errs else f"{'-':>9s}"))
        print(f"{name:14s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
