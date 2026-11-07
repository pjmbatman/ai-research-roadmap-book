"""21.5 실습: 같은 모델에 세 가지 attribution 방법을 적용해 비교하고, 개입 실험으로 검증한다.

실행: uv run --project labs python labs/reliability/interpretability/attribution_vs_intervention.py

합성 시뮬레이터(화학 반응을 흉내 낸 것으로, 실제 반응 모델이 아니다). 입력은 모두 표준화된 단위:
  temp     반응 온도            원인: 수율에 비선형으로 영향
  cat      촉매량              원인: 수율에 선형으로 영향
  humid    습도                원인이 아님. 온도와 상관 0.9 (같은 계절에 높다)
  sensor   생성물 농도 센서      결과: 수율의 하류 측정 (수율 + 잡음)
  yield    수율(목표)
모델: scikit-learn MLPRegressor. gradient는 중심 차분으로 계산한다.
방법: (1) 순열 중요도, (2) |gradient×input| 평균 (기준 0 = 데이터 평균), (3) |integrated gradients| 평균 (기준 = 데이터 평균)
검증: 시뮬레이터에서 각 입력을 +1만큼 개입(do)했을 때 실제 수율 변화와 모델 예측 변화.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

rng = np.random.default_rng(0)
NAMES = ["temp", "cat", "humid", "sensor"]


def simulate(n, do=None, shift=0.0, seed=None):
    r = rng if seed is None else np.random.default_rng(seed)   # 같은 seed = 같은 잡음(공통 난수)
    temp = r.normal(size=n)
    humid = 0.9 * temp + np.sqrt(1 - 0.81) * r.normal(size=n)
    cat = r.normal(size=n)
    if do == "temp":
        temp = temp + shift
    if do == "humid":
        humid = humid + shift
    if do == "cat":
        cat = cat + shift
    yld = 1.5 * np.tanh(temp) + 0.8 * cat + 0.3 * r.normal(size=n)
    sensor = yld + 0.2 * r.normal(size=n)
    if do == "sensor":
        sensor = sensor + shift
    return np.column_stack([temp, cat, humid, sensor]), yld


X, y = simulate(4000)
model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0).fit(X, y)
Xe, ye = simulate(2000)
base_mse = np.mean((model.predict(Xe) - ye) ** 2)
mu = X.mean(axis=0)


def grad(A, h=1e-3):                                  # A: (N, D) -> (N, D)
    G = np.zeros_like(A)
    for j in range(A.shape[1]):
        E = np.zeros(A.shape[1]); E[j] = h
        G[:, j] = (model.predict(A + E) - model.predict(A - E)) / (2 * h)
    return G


def perm_imp(j, reps=5):
    out = []
    for _ in range(reps):
        Xp = Xe.copy(); Xp[:, j] = Xe[rng.permutation(len(Xe)), j]
        out.append(np.mean((model.predict(Xp) - ye) ** 2) - base_mse)
    return np.mean(out)


gxi = np.abs((Xe - mu) * grad(Xe)).mean(axis=0)
alphas = (np.arange(20) + 0.5) / 20
ig = np.abs((Xe - mu) * np.mean([grad(mu + a * (Xe - mu)) for a in alphas], axis=0)).mean(axis=0)

print(f"모델 test MSE {base_mse:.3f} (센서가 수율을 직접 재므로 수율 잡음 분산 0.09보다 작다)\n")
print(f"{'입력':8s} {'순열중요도':>9s} {'|grad×input|':>12s} {'|IG|':>7s}   {'개입: 실제 Δy':>12s} {'모델 Δŷ':>8s}")
for j, name in enumerate(NAMES):
    Xd, yd = simulate(20000, do=name, shift=1.0, seed=1)   # 개입한 세계
    X0, y0 = simulate(20000, seed=1)                        # 같은 잡음, 개입 없음
    true_eff = yd.mean() - y0.mean()
    model_eff = model.predict(Xd).mean() - model.predict(X0).mean()
    print(f"{name:8s} {perm_imp(j):9.3f} {gxi[j]:12.3f} {ig[j]:7.3f}   {true_eff:12.3f} {model_eff:8.3f}")
