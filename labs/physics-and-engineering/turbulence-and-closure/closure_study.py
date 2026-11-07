"""26.3 실습: 학습 closure의 a priori 오차와 a posteriori 결과 비교.

실행: uv run --project labs python labs/physics-and-engineering/turbulence-and-closure/closure_study.py
- 본문과 같은 1차원 Burgers 설정(고해상도 256칸 → 거친 격자 32칸, 블록 평균 필터)
- closure: Smagorinsky(C >= 0), MLP(5점 스텐실 -> tau), MLP + 소산 제약(nu_t = softplus 꼴로 >= 0)
- 학습 nu = 0.04, 시험 nu = 0.04 / 0.02, 긴 시간(T = 6) 적분에서 발산 여부와 오차를 본다.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

L, Nf, Nc = 2 * np.pi, 256, 32
K, D = Nf // Nc, L / Nc


def ic(seed, N):
    rng, x = np.random.default_rng(seed), np.arange(N) * L / N
    return sum(rng.normal() / k * np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))


def rhs(u, dx, nu, tau=None):
    ur = np.roll(u, -1)
    F = (u**2 + u * ur + ur**2) / 6 - nu * (ur - u) / dx
    if tau is not None:
        F = F + 0.5 * (tau + np.roll(tau, -1))
    return -(F - np.roll(F, 1)) / dx


def run(u, dx, nu, model=None, T=2.0, nsave=20):
    n = int(np.ceil(T / (0.3 * min(dx / 3, dx**2 / (2 * nu + 1e-12)))))
    out = []
    for i in range(n):
        u1 = u + T / n * rhs(u, dx, nu, model(u) if model else None)
        u = 0.5 * (u + u1 + T / n * rhs(u1, dx, nu, model(u1) if model else None))
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 50:
            return None, (i + 1) * T / n
        if (i + 1) % (n // nsave) == 0:
            out.append(u)
    return np.array(out)[:nsave], T


filt = lambda u: u.reshape(*u.shape[:-1], -1, K).mean(-1)
tau_of = lambda U: 0.5 * (filt(U**2) - filt(U) ** 2)
ux = lambda ub: (np.roll(ub, -1, -1) - np.roll(ub, 1, -1)) / (2 * D)
stencil = lambda ub: np.stack([np.roll(ub, k, -1) for k in (2, 1, 0, -1, -2)], -1)   # (..., Nc, 5)


def main():
    train = {s: run(ic(s, Nf), L / Nf, 0.04, nsave=40)[0] for s in range(8)}
    Ub = np.concatenate([filt(U) for U in train.values()])
    Tt = np.concatenate([tau_of(U) for U in train.values()])
    X, y = stencil(Ub).reshape(-1, 5), Tt.ravel()

    f = (-D**2 * np.abs(ux(Ub)) * ux(Ub)).ravel()
    C = max(f @ y / (f @ f), 0.0)
    smag = lambda ub: -C * D**2 * np.abs(ux(ub)) * ux(ub)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ys = y.std()                                   # 목표값을 정규화해 학습한다
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0).fit(X, y / ys)
        # 소산 제약: tau = -nu_t * u_x, nu_t >= 0. nu_t의 목표값을 0 이상으로 자른 뒤 학습하고, 예측도 0 이상으로 자른다.
        g = -ux(Ub).ravel()
        mask = np.abs(g) > 0.2
        nut = np.clip(y[mask] / g[mask], 0, None)
        ns = nut.std()
        mlp_nut = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0).fit(X[mask], nut / ns)
    mlp_model = lambda ub: ys * mlp.predict(stencil(ub).reshape(-1, 5)).reshape(ub.shape)
    nut_model = lambda ub: -ns * np.clip(mlp_nut.predict(stencil(ub).reshape(-1, 5)).reshape(ub.shape), 0, None) * ux(ub)
    models = {"모델 없음": None, "Smagorinsky": smag, "MLP": mlp_model, "MLP + 소산 제약": nut_model}

    test = {s: run(ic(s, Nf), L / Nf, 0.04, nsave=40)[0] for s in (10, 11, 12)}
    Ub_te = np.concatenate([filt(U) for U in test.values()])
    T_te = np.concatenate([tau_of(U) for U in test.values()])
    print("[a priori] 시험 스냅샷 120개에서 tau 예측")
    for name, m in models.items():
        if m is None:
            continue
        p = m(Ub_te)
        print(f"  {name:14s} 상대 오차 {np.linalg.norm(p - T_te) / np.linalg.norm(T_te):.3f}  "
              f"상관 {np.corrcoef(p.ravel(), T_te.ravel())[0, 1]:.3f}")

    for nu, T in [(0.04, 2.0), (0.04, 6.0), (0.02, 2.0), (0.02, 6.0)]:
        refs = {s: filt(run(ic(s, Nf), L / Nf, nu, T=T)[0]) for s in (10, 11, 12)}
        print(f"[a posteriori] nu = {nu}, T = {T}")
        for name, m in models.items():
            errs, blown = [], []
            for s, ref in refs.items():
                les, t_end = run(filt(ic(s, Nf)), D, nu, m, T=T)
                if les is None:
                    blown.append(f"시드 {s}: t = {t_end:.2f}에서 발산")
                    continue
                errs.append(np.linalg.norm(les[-1] - ref[-1]) / np.linalg.norm(ref[-1]))
            msg = f"평균 상대 오차 {np.mean(errs):.3f}" if errs else "-"
            print(f"  {name:14s} {msg}" + (f"  {'; '.join(blown)}" if blown else ""))


if __name__ == "__main__":
    main()
