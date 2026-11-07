"""24.5 실습: 데이터 전용·물리 전용·하이브리드 모델을 같은 조건에서 비교한다.

본문 cell 1–2의 비교를 시드 20개로 반복하고, 참 과정의 구조가 물리 모델과 다른 경우(고온에서 두 번째 반응 경로가
켜지는 경우)를 더한다. 모든 모델은 같은 학습 데이터(300–340 K, 15점)와 같은 시험점(내삽 305–335 K, 외삽 380–420 K)을 쓴다.

실행: uv run --project labs python labs/scientific-ml/hybrid-and-surrogate-models/hybrid_compare.py
"""
import numpy as np

R_J = 8.314  # 기체 상수, J/(mol K)


def ln_rate(temp_k, ln_a, ea_j):
    return ln_a - ea_j / (R_J * temp_k)


def gp_mean(x_tr, y_tr, x_te, ell=15.0, sn=0.05):
    """평균을 데이터 평균으로 둔 RBF 커널 GP의 사후 평균(본문 cell 1의 gp와 같은 식)."""
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    alpha = np.linalg.solve(k(x_tr, x_tr) + sn**2 * np.eye(x_tr.size), y_tr - y_tr.mean())
    return y_tr.mean() + k(x_te, x_tr) @ alpha


SCENARIOS = {
    # 참 과정의 구조가 물리 모델(단일 Arrhenius)과 같다
    "구조 일치": lambda t: ln_rate(t, 18.0, 60e3),
    # 고온에서 활성화 에너지가 큰 두 번째 경로가 켜진다(학습 범위에서는 거의 보이지 않는다)
    "구조 불일치": lambda t: np.logaddexp(ln_rate(t, 18.0, 60e3), ln_rate(t, 40.0, 130e3)),
}
physics = lambda t: ln_rate(t, 16.0, 55e3)  # 문헌값(파라미터가 틀림)


def compare(truth, seed):
    rng = np.random.default_rng(seed)
    t_tr = np.sort(rng.uniform(300, 340, 15))
    y_tr = truth(t_tr) + 0.05 * rng.normal(size=15)
    t_in, t_out = np.linspace(305, 335, 50), np.linspace(380, 420, 50)
    X = np.column_stack([np.ones(15), -1 / (R_J * t_tr)])
    ln_a, ea = np.linalg.lstsq(X, y_tr, rcond=None)[0]
    resid = y_tr - physics(t_tr)
    models = [
        lambda t: gp_mean(t_tr, y_tr, t),
        physics,
        lambda t: physics(t) + gp_mean(t_tr, resid, t),
        lambda t: ln_rate(t, ln_a, ea),
    ]
    rmse = lambda f, t: np.sqrt(np.mean((f(t) - truth(t)) ** 2))
    return np.array([[rmse(f, t_in), rmse(f, t_out)] for f in models])  # (4, 2)


if __name__ == "__main__":
    names = ["데이터 전용 GP", "물리 전용(문헌값)", "잔차 학습 하이브리드", "파라미터 학습"]
    for scen, truth in SCENARIOS.items():
        res = np.array([compare(truth, s) for s in range(20)])  # (20, 4, 2)
        gap = truth(400.0) - ln_rate(400.0, 18.0, 60e3)
        print(f"[{scen}] 시드 20개 평균 RMSE (ln k), 400 K에서 두 번째 경로의 기여 {gap:.2f}")
        for i, name in enumerate(names):
            m = res[:, i].mean(axis=0)
            print(f"  {name:<12}  내삽 {m[0]:.3f}   외삽 {m[1]:.3f}")
