"""10.5 실습: 감쇠 진동자의 파라미터 추정을 Adam · L-BFGS · Levenberg–Marquardt로 비교한다.

모델  x(t) = A exp(-gamma t) cos(omega t + phi)
  A      진폭 [m]
  gamma  감쇠율 [1/s]
  omega  각진동수 [rad/s]
  phi    위상 [rad]

1) 같은 시작점에서 세 방법이 손실을 줄이는 속도(gradient 또는 Jacobian 평가 횟수 기준)와 최종 정확도
2) 시작 각진동수에 따른 수렴 결과 (비볼록성)
3) mini-batch로 L-BFGS를 돌리면 생기는 일

실행: uv run --project labs python labs/optimization/second-order-methods/param_estimation.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

OUT = Path(__file__).parent / "outputs"

rng = np.random.default_rng(0)
t_s = np.linspace(0.0, 10.0, 200)                           # (N,) 측정 시각 [s]
THETA_TRUE = np.array([1.0, 0.3, 2.0, 0.5])                 # A [m], gamma [1/s], omega [rad/s], phi [rad]
NOISE_M = 0.05                                              # 측정 잡음 표준편차 [m]


def model(th, t=t_s):
    A, gam, om, ph = th
    return A * np.exp(-gam * t) * np.cos(om * t + ph)


def jacobian(th, t=t_s):
    """d model / d theta. 반환 (N, 4)"""
    A, gam, om, ph = th
    e, c, s = np.exp(-gam * t), np.cos(om * t + ph), np.sin(om * t + ph)
    return np.stack([e * c, -A * t * e * c, -A * t * e * s, -A * e * s], axis=1)


x_obs = model(THETA_TRUE) + NOISE_M * rng.normal(size=t_s.size)


def resid(th, idx=slice(None)):
    return model(th, t_s[idx]) - x_obs[idx]


def loss(th, idx=slice(None)):                              # 평균 제곱 오차의 절반
    r = resid(th, idx)
    return 0.5 * np.mean(r**2)


def grad(th, idx=slice(None)):
    return jacobian(th, t_s[idx]).T @ resid(th, idx) / len(t_s[idx])


# ---------------------------------------------------------------- 1) 세 방법 비교
def run_adam(th0, lr, steps=5000, b1=0.9, b2=0.999, eps=1e-8):
    th, m, v, hist = np.array(th0, float), np.zeros(4), np.zeros(4), []
    for t in range(1, steps + 1):
        g = grad(th)
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        hist.append(loss(th))
    return th, np.array(hist)


def run_lbfgs(th0):
    hist = []
    def fun(th):
        val = loss(th)
        hist.append(val)
        return val, grad(th)
    res = minimize(fun, th0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-12})
    return res.x, np.minimum.accumulate(hist)


def run_lm(th0):
    hist = []
    def fun(th):
        r = resid(th)
        hist.append(0.5 * np.mean(r**2))
        return r
    res = least_squares(fun, th0, jac=lambda th: jacobian(th), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return res.x, np.minimum.accumulate(hist), res


def evals_to(hist, target):
    hit = np.nonzero(hist <= target)[0]
    return int(hit[0]) + 1 if hit.size else None


def part1(th0):
    print(f"1) 같은 시작점 theta0 = {th0}에서의 비교 (참값 {THETA_TRUE}, 잡음 {NOISE_M} m)")
    th_lm, h_lm, res_lm = run_lm(th0)
    best = h_lm[-1]
    th_lb, h_lb = run_lbfgs(th0)
    runs = {"LM": (th_lm, h_lm), "L-BFGS": (th_lb, h_lb)}
    for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        runs[f"Adam lr={lr:g}"] = run_adam(th0, lr)
    print(f"   최소 손실(LM) = {best:.6e}")
    print("   방법             손실이 최소의 (1 + 1e-3)배 이내  (1 + 1e-6)배 이내   5000회/종료 후 손실   |theta - 참값|")
    for name, (th, h) in runs.items():
        e3, e6 = evals_to(h, best * (1 + 1e-3)), evals_to(h, best * (1 + 1e-6))
        fmt = lambda e: f"{e:5d}회" if e else "  도달 못함"
        print(f"   {name:15s}  {fmt(e3):>22s}  {fmt(e6):>14s}  {h[-1]:18.6e}   {np.linalg.norm(th - THETA_TRUE):.4f}")
    # 표준오차: 잔차 분산 s^2 (J^T J)^{-1}
    J = jacobian(th_lm)
    dof = t_s.size - 4
    s2 = res_lm.fun @ res_lm.fun / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(J.T @ J)))
    print("   LM 추정값        :", np.round(th_lm, 4))
    print("   표준오차(J^T J)  :", np.round(se, 4), f" (잔차 표준편차 추정 {np.sqrt(s2):.4f} m)")
    return runs


# ---------------------------------------------------------------- 2) 시작 각진동수
def part2():
    print("\n2) 시작 각진동수 omega0에 따른 결과 (다른 파라미터는 theta0와 같다)")
    def kind(th):
        if abs(th[2] - THETA_TRUE[2]) > 0.05:
            return "참값과 다른 점에서 멈춤"
        return "참값 근처" if th[0] > 0 else "동등한 해(-A, phi - pi)"
    for om0 in [0.5, 1.0, 3.0, 6.0, 8.0, 20.0]:
        th0 = np.array([0.5, 0.1, om0, 0.0])
        with np.errstate(all="ignore"):
            th_lm, th_lb = run_lm(th0)[0], run_lbfgs(th0)[0]
        for name, th in [("LM", th_lm), ("L-BFGS", th_lb)]:
            head = f"omega0 = {om0:4.1f}" if name == "LM" else ""
            print(f"   {head:13s} {name:6s} -> {np.round(th, 3)}  손실 {loss(th):.3e}  |grad| {np.linalg.norm(grad(th)):.1e}  {kind(th)}")


# ---------------------------------------------------------------- 3) mini-batch L-BFGS
def lbfgs_direction(g, S, Y):
    """two-loop recursion: 근사 역Hessian과 g의 곱의 음수를 반환"""
    q, alphas = g.copy(), []
    for s, y in reversed(list(zip(S, Y))):
        a = (s @ q) / (y @ s)
        alphas.append(a)
        q -= a * y
    if S:
        q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
    for (s, y), a in zip(zip(S, Y), reversed(alphas)):
        q += (a - (y @ q) / (y @ s)) * s
    return -q


def my_lbfgs(th0, batch, iters, m=10, seed=0, record=()):
    """batch=None이면 전체 데이터. 곡률 쌍 (s, y)는 같은 batch에서 계산한다."""
    r = np.random.default_rng(seed)
    th, S, Y, out = np.array(th0, float), [], [], {}
    for k in range(1, iters + 1):
        idx = r.choice(t_s.size, size=batch, replace=False) if batch else slice(None)
        g = grad(th, idx)
        d = lbfgs_direction(g, S, Y)
        if g @ d >= 0:                                      # 내려가는 방향이 아니면 기억을 버린다
            S, Y, d = [], [], -g
        step = 1.0
        while loss(th + step * d, idx) > loss(th, idx) + 1e-4 * step * g @ d and step > 1e-10:
            step /= 2                                       # 같은 batch 위에서의 Armijo backtracking
        new = th + step * d
        s, y = new - th, grad(new, idx) - g
        if s @ y > 1e-12:
            S, Y = (S + [s])[-m:], (Y + [y])[-m:]
        th = new
        if k in record:
            out[k] = loss(th)
    return out


def adam_minibatch(th0, batch, iters, lr0, seed=0, record=()):
    r = np.random.default_rng(seed)
    th, m, v, out = np.array(th0, float), np.zeros(4), np.zeros(4), {}
    for t in range(1, iters + 1):
        g = grad(th, r.choice(t_s.size, size=batch, replace=False))
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        lr = lr0 / (1 + t / 100)                             # 감소하는 학습률
        th = th - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        if t in record:
            out[t] = loss(th)
    return out


def part3(th0, best):
    rec = (30, 300, 3000)
    print("\n3) mini-batch로 L-BFGS를 돌리면: 전체 손실의 초과분 (손실 - 최소 손실), 시드 5개 중앙값")
    print("   설정                         " + "".join(f"{k:>8d}회" for k in rec))
    for name, fn in [("L-BFGS 전체 batch", lambda sd: my_lbfgs(th0, None, 3000, seed=sd, record=rec)),
                     ("L-BFGS batch 50", lambda sd: my_lbfgs(th0, 50, 3000, seed=sd, record=rec)),
                     ("L-BFGS batch 10", lambda sd: my_lbfgs(th0, 10, 3000, seed=sd, record=rec)),
                     ("Adam batch 10, lr 감소", lambda sd: adam_minibatch(th0, 10, 3000, 0.03, seed=sd, record=rec))]:
        res = [fn(sd) for sd in range(5)]
        med = [np.median([r_[k] - best for r_ in res]) for k in rec]
        print(f"   {name:24s}" + "".join(f"{x:9.1e}" for x in med))


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    best = min(h.min() for _, h in runs.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (_, h) in runs.items():
        ax.loglog(np.arange(1, len(h) + 1), h - best + 1e-16, label=name)
    ax.set_xlabel("gradient / Jacobian evaluations")
    ax.set_ylabel("loss - best")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=120)


if __name__ == "__main__":
    th0 = np.array([0.5, 0.1, 1.8, 0.0])
    runs = part1(th0)
    part2()
    part3(th0, min(h.min() for _, h in runs.values()))
    plot(runs)
