"""11.7 실습: Euler–Maruyama로 Ornstein–Uhlenbeck 과정을 시뮬레이션하고 정상분포를 확인한다.

dX = -theta X dt + sigma dW 의 정상분포는 N(0, sigma^2 / (2 theta)), 자기상관은 exp(-theta tau)다.
시간 간격 dt를 바꿔 가며 두 이산화를 비교한다.

- Euler–Maruyama(EM): x_{k+1} = (1 - theta dt) x_k + sigma sqrt(dt) xi
- 정확한 이산화:      x_{k+1} = exp(-theta dt) x_k + sigma sqrt((1 - exp(-2 theta dt)) / (2 theta)) xi

EM 갱신은 계수 phi = 1 - theta dt인 AR(1)이므로 정상 분산은 sigma^2 dt / (1 - phi^2)
= sigma^2 / (theta (2 - theta dt))이고, theta dt >= 2이면 발산한다.

실행: uv run --project labs python labs/probability/stochastic-processes/ou_euler_maruyama.py
"""
import numpy as np

SEED = 0
THETA, SIGMA = 1.0, np.sqrt(2.0)          # 정상 분산 sigma^2 / (2 theta) = 1, 상관시간 1/theta = 1
T_BURN, T_RUN = 20.0, 200.0               # 정상 상태에 도달할 때까지 버리는 시간, 통계를 모으는 시간
N_PATHS = 200
DTS = (0.01, 0.1, 0.5, 1.0, 1.5, 1.9, 2.1)


def simulate(dt, exact, rng):
    phi = np.exp(-THETA * dt) if exact else 1 - THETA * dt
    noise = SIGMA * np.sqrt((1 - np.exp(-2 * THETA * dt)) / (2 * THETA)) if exact else SIGMA * np.sqrt(dt)
    n_burn, n_run = int(round(T_BURN / dt)), int(round(T_RUN / dt))
    x = np.zeros(N_PATHS)                                   # (N_PATHS,)
    out = np.empty((n_run, N_PATHS))                        # (n_run, N_PATHS)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n_burn + n_run):
            x = phi * x + noise * rng.normal(size=N_PATHS)
            if k >= n_burn:
                out[k - n_burn] = x
    return out


def lag1_corr(xs):
    """한 step 떨어진 값 사이의 상관. 참 OU에서는 exp(-theta dt)다."""
    return np.corrcoef(xs[:-1].ravel(), xs[1:].ravel())[0, 1]


def main():
    rng = np.random.default_rng(SEED)
    print(f"OU: theta = {THETA}, sigma^2 = {SIGMA**2:.1f}, 참 정상 분산 1, 한 step 자기상관의 참값 exp(-theta dt)")
    print(f"(경로 {N_PATHS}개, 시간 {T_BURN:.0f}를 버리고 {T_RUN:.0f} 동안 수집, 시드 {SEED})")
    print("  dt   | EM 분산  이론     자기상관 | 정확 이산화 분산  자기상관 | 참 자기상관")
    for dt in DTS:
        em = simulate(dt, exact=False, rng=rng)
        ex = simulate(dt, exact=True, rng=rng)
        theory = SIGMA**2 / (THETA * (2 - THETA * dt)) if THETA * dt < 2 else np.inf
        em_var = em.var()
        em_s = f"{em_var:8.3f}" if np.isfinite(em_var) and em_var < 1e6 else "    발산"
        th_s = f"{theory:8.3f}" if np.isfinite(theory) else "    발산"
        ac_s = f"{lag1_corr(em):8.3f}" if np.isfinite(em_var) and em_var < 1e6 else "       -"
        print(f"{dt:5.2f}  |{em_s} {th_s} {ac_s} |    {ex.var():8.3f}     {lag1_corr(ex):8.3f} |  {np.exp(-THETA * dt):.3f}")


if __name__ == "__main__":
    main()
