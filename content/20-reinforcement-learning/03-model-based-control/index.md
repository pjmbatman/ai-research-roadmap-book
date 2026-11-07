---
id: reinforcement-learning/model-based-control
title: "모델 기반 방법과 최적제어"
title_en: "Model-Based Methods and Optimal Control"
tier: advanced
status: draft
prerequisites: [reinforcement-learning/mdp-and-dp, optimization/constrained-optimization]
---

# 모델 기반 방법과 최적제어

20.2장의 model-free 방법은 환경의 모형 없이 경험만으로 배운다. 그 대가는 표본이다. 단순한 연속 제어 과제에도 수십만에서 수백만 step의 상호작용이 필요한 경우가 흔하다. 실험 한 번에 몇 시간이 걸리는 장비, 고장 나면 수리비가 드는 로봇, 한 번 실행에 GPU 몇 시간이 드는 시뮬레이터에서는 이 표본 수를 감당할 수 없다. 반면 과학과 공학에는 **동역학 모형이 이미 있거나 데이터로 배울 수 있는** 문제가 많다. 운동 방정식, 반응 속도식, 회로 방정식이 그렇다.

모형이 있으면 시행착오 대신 **계획**(planning)을 할 수 있다. 이 장은 제어 공학이 수십 년 동안 다듬어 온 최적제어(LQR, MPC)를 20.1장의 동적계획법 언어로 다시 읽고, 모형을 데이터로 학습해 계획하는 모델 기반 RL과, 시뮬레이터를 미분해 정책을 최적화하는 방법으로 넘어간다. 핵심 질문은 하나다. **모형이 틀렸을 때 계획은 얼마나 틀리는가.**

## 학습 목표

- 최적제어와 강화학습의 관계를 이해한다
- MPC를 구현한다

## 최적제어

**왜 필요한가.** 동역학을 아는 시스템을 원하는 상태로 보내는 입력을 찾는 문제는 RL보다 훨씬 오래된 분야인 최적제어(optimal control)의 주제다. 용어는 다르지만 같은 문제를 푼다. 상태 $\mathbf{x}$, 입력(행동) $\mathbf{u}$, 동역학 $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$, 그리고 최소화할 **비용** $\sum_t c(\mathbf{x}_t, \mathbf{u}_t)$가 있다. 비용은 보상의 음수다. 제어 이론은 연속 상태·연속 입력과 결정적 동역학을 기본으로 하고, RL은 이산 상태와 확률적 전이에서 출발했다는 차이가 있을 뿐이다.

**LQR.** 동역학이 선형이고 비용이 이차식인 경우가 **선형-이차 조절기**(linear-quadratic regulator, LQR) 문제다.

$$
\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t, \qquad
J = \sum_{t=0}^{\infty} \big(\mathbf{x}_t^\top \mathbf{Q}\mathbf{x}_t + \mathbf{u}_t^\top \mathbf{R}\mathbf{u}_t\big).
$$

$\mathbf{Q} \succeq 0$은 상태가 목표(원점)에서 벗어난 정도의 비용, $\mathbf{R} \succ 0$은 입력을 쓰는 비용이다. 두 행렬의 상대적 크기가 "얼마나 빨리 가고 싶은가"와 "얼마나 아껴 쓰고 싶은가"의 균형을 정한다.

**Bellman 방정식에서 Riccati 방정식으로.** 이 문제에 20.1장의 동적계획법을 적용한다. 가치함수(여기서는 최소 비용)를 이차식 $V(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}\mathbf{x}$로 가정하고 Bellman 방정식에 넣으면

$$
\mathbf{x}^\top \mathbf{P}\mathbf{x} = \min_{\mathbf{u}} \Big[\mathbf{x}^\top \mathbf{Q}\mathbf{x} + \mathbf{u}^\top \mathbf{R}\mathbf{u} + (\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u})^\top \mathbf{P}(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u})\Big]
$$

이다. 괄호 안은 $\mathbf{u}$의 볼록 이차식이므로 기울기를 0으로 두면(10.1장) 최적 입력이 **상태의 선형 함수**로 나온다.

$$
\mathbf{u}^* = -\mathbf{K}\mathbf{x}, \qquad \mathbf{K} = (\mathbf{R} + \mathbf{B}^\top \mathbf{P}\mathbf{B})^{-1}\mathbf{B}^\top \mathbf{P}\mathbf{A}.
$$

이를 다시 넣으면 $\mathbf{P}$에 대한 **이산 대수 Riccati 방정식**(discrete algebraic Riccati equation, DARE)을 얻는다.

$$
\mathbf{P} = \mathbf{Q} + \mathbf{A}^\top \mathbf{P}\mathbf{A} - \mathbf{A}^\top \mathbf{P}\mathbf{B}(\mathbf{R} + \mathbf{B}^\top \mathbf{P}\mathbf{B})^{-1}\mathbf{B}^\top \mathbf{P}\mathbf{A}.
$$

Bellman 방정식의 $\max$(여기서는 $\min$)가 닫힌 식으로 풀리고, 가치 반복이 행렬 $\mathbf{P}$의 반복이 된다. 상태가 연속인데도 가치함수 전체를 행렬 하나로 정확히 나타낼 수 있는, 연속 상태 MDP에서 거의 유일하게 정확히 풀리는 경우다.

**직관.** 1 m 떨어진 곳에 정지해 있는 물체를 가속도를 조절해 원점으로 가져오는 이중 적분기(double integrator)로 확인한다. 위치·속도가 상태, 가속도가 입력이고, 시간 간격은 0.1 s다.

```python {exec="browser" session="control"}
import numpy as np
from scipy.linalg import solve_discrete_are

dt_s = 0.1                                   # 시간 간격 [s]
A = np.array([[1.0, dt_s], [0.0, 1.0]])      # 상태 x = (위치 [m], 속도 [m/s]), (2, 2)
B = np.array([[0.5 * dt_s**2], [dt_s]])      # 입력 u = 가속도 [m/s^2], (2, 1)
Qc, Rc = np.diag([1.0, 0.1]), np.array([[0.1]])   # 상태 비용, 입력 비용
P = Qc.copy()                                # 가치함수 V(x) = x^T P x, 지평 끝에서 출발
for k in range(500):                         # Riccati 반복 = 이차 가치함수에 대한 DP
    K = np.linalg.solve(Rc + B.T @ P @ B, B.T @ P @ A)    # (1, 2), u = -K x
    P_new = Qc + A.T @ P @ (A - B @ K)
    if np.abs(P_new - P).max() < 1e-12:
        break
    P = P_new
print(f"Riccati {k + 1}번 반복, K = {K.round(3)}")
print("scipy DARE와 일치:", np.allclose(P, solve_discrete_are(A, B, Qc, Rc)))
print("폐루프 고윳값 크기:", np.abs(np.linalg.eigvals(A - B @ K)).round(3))
def rollout(policy, x0, T=60):               # 실제 시스템에서 비용과 궤적을 잰다
    x, cost, xs, us = x0.copy(), 0.0, [], []
    for t in range(T):
        u = policy(x, t)
        cost += x @ Qc @ x + u @ Rc @ u
        xs.append(x); us.append(u)
        x = A @ x + B @ u
    return cost, np.array(xs), np.array(us)
x0 = np.array([1.0, 0.0])                    # 1 m 떨어진 곳에서 정지 상태로 출발
cost, xs, us = rollout(lambda x, t: -K @ x, x0)
print(f"LQR 비용 {cost:.3f}  (예측 x0^T P x0 = {x0 @ P @ x0:.3f}),  최대 |u| = {np.abs(us).max():.2f} m/s^2,"
      f"  최대 |v| = {np.abs(xs[:, 1]).max():.2f} m/s")
```

```output
Riccati 112번 반복, K = [[2.762 2.508]]
scipy DARE와 일치: True
폐루프 고윳값 크기: [0.874 0.874]
LQR 비용 9.078  (예측 x0^T P x0 = 9.078),  최대 |u| = 2.76 m/s^2,  최대 |v| = 0.78 m/s
```

Riccati 반복(20.1장의 가치 반복과 같은 구조)이 scipy의 DARE 풀이와 같은 $\mathbf{P}$에 도달했다. 폐루프 행렬 $\mathbf{A} - \mathbf{B}\mathbf{K}$의 고윳값 크기가 1보다 작으므로 상태는 원점으로 수렴한다(7.5장). 실제로 굴려 본 비용이 $\mathbf{x}_0^\top \mathbf{P}\mathbf{x}_0$와 같다는 것은 $\mathbf{P}$가 정말 가치함수라는 뜻이다. 다만 이 제어기는 처음에 2.76 m/s²의 가속도를 쓰고, 속도가 0.78 m/s까지 올라간다. LQR에는 이것을 막을 방법이 없다.

::quiz{id="lqr-gain-scalar"}

**비용함수와 제약.** 실제 시스템에는 거의 언제나 **제약**이 있다. 모터 토크와 밸브 개도에는 한계가 있고, 반응기 온도나 배터리 셀 온도에는 넘으면 안 되는 안전 한계가 있다. LQR은 무제약 이차 문제라서 제약을 넣으면 닫힌 식이 사라진다. 입력을 한계에서 잘라(saturation) 쓰는 것은 흔한 임시방편이지만, 그렇게 잘린 제어기는 더 이상 최적이 아니고 상태 제약은 전혀 다루지 못한다. 제약을 계획 안에서 다루는 것이 다음 절의 MPC다. 제약을 비용에 넣는 방식(벌점, Lagrangian)과 그 의미는 10.2장에서 다뤘다.

**선형화.** 비선형 시스템도 동작점 근처에서 선형화하면(7.2장, 8.1장) LQR을 쓸 수 있다. 뒤집힌 진자를 세워 두는 제어기, 궤도 위성의 자세 제어가 이 방식이다. 동작점에서 멀어지면 선형화가 깨지므로, 궤적을 따라 선형화를 반복하는 방법(iterative LQR 계열)으로 확장한다.

## MPC

**왜 필요한가.** 제약이 있으면 최적 정책 $\pi^*(\mathbf{x})$를 모든 상태에 대해 미리 계산해 두기 어렵다. 그런데 지금 이 상태에서 **앞으로 몇 step의 입력열**만 최적화하는 것은 유한 차원의 최적화 문제라 풀 수 있다.

**정의.** **모델 예측 제어**(model predictive control, MPC)는 매 step 다음을 반복한다.

1. 현재 상태 $\mathbf{x}_t$를 측정한다.
2. 모형으로 $H$ step 앞까지 예측하며 유한 지평 문제를 푼다.
   $$
   \min_{\mathbf{u}_t, \dots, \mathbf{u}_{t+H-1}} \sum_{h=0}^{H-1} c(\hat{\mathbf{x}}_{t+h}, \mathbf{u}_{t+h}) \quad \text{s.t.} \quad \hat{\mathbf{x}}_{t+h+1} = f(\hat{\mathbf{x}}_{t+h}, \mathbf{u}_{t+h}),\ \ \mathbf{u} \in \mathcal{U},\ \ \hat{\mathbf{x}} \in \mathcal{X}.
   $$
3. 얻은 입력열의 **첫 입력만** 적용하고, 나머지는 버린다.

다음 step에는 지평이 한 칸 밀린 문제를 새로 푼다. 이를 **receding horizon**이라 한다. 첫 입력만 쓰는 이유는 피드백이다. 모형이 틀렸거나 외란이 들어와 실제 상태가 예측과 달라지면, 다음 step의 계획이 새로 측정한 상태에서 시작하므로 그 차이가 바로 반영된다. 계획 전체를 개루프로 실행하면 이 교정이 없다.

::quiz{id="receding-horizon"}

**풀이 방법.** 동역학이 선형이고 비용이 이차, 제약이 선형 부등식이면 매 step의 문제는 이차계획(QP)이고 10.2장의 KKT 조건을 푸는 전용 solver로 밀리초 단위에 풀 수 있다. 동역학이 비선형이거나 신경망이면 비선형 최적화가 된다. 가장 단순한 대안은 **랜덤 슈팅**(random shooting)이다. 입력열을 무작위로 $N$개 뽑아 모형으로 굴리고, 비용이 가장 낮은 것을 고른다. 미분이 필요 없어 모형이 블랙박스여도 되고, 병렬화가 쉽다. 좋은 입력열 주변으로 표집 분포를 반복해서 좁히는 교차 엔트로피 방법(cross-entropy method, CEM)은 같은 계산량으로 더 좋은 해를 찾는다.

**제약 처리.** 이중 적분기에 입력 한계 $|u| \le 1$ m/s²과 속도 한계 $|v| \le 0.5$ m/s를 둔다. 입력 한계는 표집 범위로, 속도 한계는 위반량에 큰 벌점을 주는 방식으로 넣는다. 비교 대상은 LQR의 입력을 한계에서 자른 제어기다.

```python {exec="browser" session="control"}
U_MAX, V_MAX = 1.0, 0.5                      # 입력 한계 [m/s^2], 속도 한계 [m/s]
rng = np.random.default_rng(0)

def mpc(x, t, model=lambda X, U: X @ A.T + U @ B.T, H=15, N=1000):
    U = rng.uniform(-U_MAX, U_MAX, size=(N, H, 1))         # 후보 입력열 N개, (N, H, 1)
    X, J = np.tile(x, (N, 1)), np.zeros(N)                 # (N, 2), (N,)
    for h in range(H):                       # 모델로 H step 앞까지 굴려 비용을 센다
        J += np.einsum("ni,ij,nj->n", X, Qc, X) + Rc[0, 0] * U[:, h, 0] ** 2
        J += 1e3 * np.maximum(np.abs(X[:, 1]) - V_MAX, 0.0)   # 속도 제약 위반 벌점
        X = model(X, U[:, h])
    return U[np.argmin(J), 0]                # 가장 좋은 입력열의 첫 입력만 쓴다

lqr_clip = lambda x, t: np.clip(-K @ x, -U_MAX, U_MAX)     # 비교: LQR을 한계에서 자른 것
for name, pol in [("LQR(자르기)", lqr_clip), ("MPC(랜덤 슈팅)", mpc)]:
    c, xs, us = rollout(pol, x0)
    print(f"{name:14s} 비용 {c:6.3f}  최대 |u| {np.abs(us).max():.2f}  최대 |v| {np.abs(xs[:, 1]).max():.2f}"
          f"  최종 |x| {np.abs(xs[-1, 0]):.3f} m")
```

```output
LQR(자르기)       비용 10.113  최대 |u| 1.00  최대 |v| 0.73  최종 |x| 0.001 m
MPC(랜덤 슈팅)     비용 12.088  최대 |u| 0.96  최대 |v| 0.50  최종 |x| 0.044 m
```

자른 LQR은 입력 한계는 지키지만 속도가 0.73 m/s까지 올라가 속도 제약을 어긴다. LQR은 속도 한계가 있다는 것을 모르기 때문이다. MPC는 15 step 앞을 보며 속도가 0.5 m/s를 넘지 않는 입력열만 고르므로 제약을 지킨다. 그 대가로 비용이 더 높다. 제약이 있는 문제의 최적 비용은 제약이 없는 문제보다 높을 수밖에 없다. 랜덤 슈팅은 최적해가 아닌 근사해를 주므로 최종 위치도 0.044 m 남았다. 표집 수와 지평을 늘리거나 CEM, QP solver를 쓰면 줄어든다.

**연구에서의 의미.** MPC는 화학 공정 제어에서 오래 쓰여 온 산업 표준 기법이고, 건물 냉난방, 배터리 충전, 자율 주행, 로봇 보행으로 퍼졌다. MPC의 성능은 거의 전적으로 **모형의 질**과 **매 step 최적화를 제시간에 끝낼 수 있는지**에 달려 있다. 제어 주기가 10 ms이면 그 안에 계획이 끝나야 한다. 학습된 신경망 모형을 MPC에 쓰려면 이 계산 시간도 함께 설계해야 한다.

:::note[분야 예시]
**배터리 고속 충전**: 전류를 크게 하면 빨리 충전되지만 셀 온도와 리튬 석출 위험이 커진다. 전기화학 모형으로 앞으로의 온도와 전압을 예측하며 "충전 시간을 최소화하되 온도·전압 한계를 넘지 않는" 전류를 매 step 고르는 것이 MPC 문제다. **생물 반응기**: 배지 공급 속도를 조절해 세포 농도를 키우되 용존 산소와 대사 부산물 농도의 한계를 지키는 유가식(fed-batch) 배양 제어도 같은 구조다. **실험 설계**: 다음 측정 조건을 고르는 일도 "모형(대리모델)으로 앞을 예측하고, 지금 할 실험 하나만 정한 뒤, 결과를 보고 다시 계획한다"는 점에서 receding horizon과 같은 구조다(25.3장, 25.4장).
:::

## 모델 기반 RL

**왜 필요한가.** MPC는 정확한 모형을 가정한다. 모형이 없으면 데이터로 배울 수 있다. 상호작용 데이터 $(\mathbf{x}, \mathbf{u}, \mathbf{x}')$로 동역학 모형 $\hat{f}_\phi$를 지도학습으로 적합하고, 그 모형으로 계획하거나(MPC) 모형 안에서 가상의 경험을 만들어 정책을 학습하는 것이 **모델 기반 RL**이다. 모형을 한 번 배우면 같은 데이터에서 훨씬 많은 것을 뽑아낼 수 있어, model-free 방법보다 표본 효율이 크게 좋은 경우가 많다.

**모델 오차의 누적.** 학습된 모형의 1 step 예측 오차가 작아도, 계획은 모형을 **자기 예측 위에서** 여러 step 굴린다. 1 step의 오차가 다음 step의 입력이 되고, 그 오차가 동역학을 따라 증폭된다. 진자(비선형, $\ddot\theta = -(g/\ell)\sin\theta - b\dot\theta + u$)의 데이터 500개로 선형 회귀 모형을 적합하고, 모형을 굴린 예측이 실제와 얼마나 벌어지는지 본다.

```python {exec="browser" session="control"}
g_ms2, len_m, damp = 9.81, 1.0, 0.1          # 진자: 중력 가속도, 길이, 감쇠
def pend(X, U):                              # 실제 시스템(비선형), X = (각도 [rad], 각속도 [rad/s])
    th, w = X[..., 0], X[..., 1]
    w2 = w + dt_s * (-g_ms2 / len_m * np.sin(th) - damp * w + U[..., 0])
    return np.stack([th + dt_s * w2, w2], axis=-1)

X_data = rng.uniform([-1.5, -2.0], [1.5, 2.0], size=(500, 2))   # 무작위 상태 500개
U_data = rng.uniform(-2.0, 2.0, size=(500, 1))                  # 무작위 토크 [1/s^2]
Y = pend(X_data, U_data) + rng.normal(0, 0.01, size=(500, 2))   # 측정 잡음
Z = np.hstack([X_data, U_data, np.ones((500, 1))])              # (500, 4) 선형 모델 특징
Theta, *_ = np.linalg.lstsq(Z, Y, rcond=None)                    # (4, 2) 최소제곱 적합
learned = lambda X, U: np.hstack([X, U, np.ones((len(X), 1))]) @ Theta

X0 = rng.uniform([-1.2, -1.0], [1.2, 1.0], size=(200, 2))   # 시험 초기 상태 200개, 입력 0
x_true, x_mod, zero, worst = X0, X0, np.zeros((200, 1)), np.zeros(200)
for k in range(1, 41):                       # 모델을 자기 예측 위에서 굴린다(open-loop rollout)
    x_true, x_mod = pend(x_true, zero), learned(x_mod, zero)
    worst = np.maximum(worst, np.abs(x_mod[:, 0] - x_true[:, 0]))   # 지평 k까지의 최대 오차
    if k in (1, 5, 10, 20, 40):
        print(f"지평 {k:2d} step 안의 최대 각도 오차(200개 평균) {worst.mean():.3f} rad")
```

```output
지평  1 step 안의 최대 각도 오차(200개 평균) 0.005 rad
지평  5 step 안의 최대 각도 오차(200개 평균) 0.059 rad
지평 10 step 안의 최대 각도 오차(200개 평균) 0.083 rad
지평 20 step 안의 최대 각도 오차(200개 평균) 0.207 rad
지평 40 step 안의 최대 각도 오차(200개 평균) 0.468 rad
```

학습 데이터에서 1 step 각도 오차는 0.005 rad 수준이지만, 모형을 40 step(4 s) 굴리면 최대 오차가 평균 0.47 rad까지 커진다. 1 step 오차로 모형을 평가하면 계획에 쓸 수 있는지 알 수 없다. 오차가 커지는 이유는 두 가지다. 선형 모형은 $\sin\theta$의 휘어짐을 표현하지 못해 **체계적 편향**이 있고, 그 편향이 진동의 위상을 조금씩 어긋나게 하며 매 step 쌓인다.

::quiz{id="model-error-compounding"}

**모델 오차에 대응하는 방법.** 연구에서 쓰는 처방은 다음과 같다.

- **짧은 지평과 잦은 재계획.** MPC의 receding horizon 자체가 오차 누적을 제한한다. 지평이 짧을수록 모형 오차의 영향이 작고, 대신 먼 미래를 보지 못한다.
- **불확실성을 아는 모형.** 신경망 모형 여러 개(ensemble)의 예측이 갈리는 곳은 데이터가 부족한 곳이다. 확률적 모형의 ensemble로 불확실성을 전파하며 계획하면, 적은 시행으로 model-free 방법에 가까운 성능을 낼 수 있다는 결과가 있다(Chua 외, 2018). 불확실한 영역으로 가는 계획에 벌점을 주면 모형이 틀릴 곳을 피한다(20.4장의 보수적 방법과 같은 생각이다).
- **물리 구조를 모형에 넣기.** 운동 방정식의 형태를 알면 그 구조를 모형에 넣는다(24.5장). 실습에서 특징에 $\sin\theta$ 하나를 넣는 것만으로 결과가 달라진다.
- **데이터를 계획이 가는 곳에서 모으기.** 계획한 궤적을 실제로 실행해 데이터를 추가하고 모형을 다시 학습하는 반복이 필요하다. 처음 데이터가 덮지 않은 영역으로 계획이 가면, 그곳에서 모형은 근거 없이 낙관적일 수 있다.

**Dyna.** 모형을 계획에 쓰는 대신 모형으로 가상 경험을 생성해 20.2장의 Q-learning 같은 model-free 학습에 섞어 쓸 수도 있다. Sutton·Barto 교재 8장의 Dyna 구조가 이 생각의 원형이다. 실제 경험 하나마다 모형에서 가상 경험 여러 개를 만들어 같은 갱신을 하므로 표본 효율이 오른다. 대신 가상 경험의 오차가 그대로 가치함수에 들어간다.

## 미분가능 시뮬레이션과 제어

**왜 필요한가.** 랜덤 슈팅은 입력열의 차원이 커지면 좋은 후보를 뽑을 확률이 급격히 줄어든다. 시뮬레이터가 미분 가능하면 비용의 입력에 대한 **기울기**를 계산해 경사하강으로 입력열이나 정책 파라미터를 최적화할 수 있다. 차원이 수천이어도 기울기 한 번의 비용은 시뮬레이션 한 번의 몇 배에 불과하다.

**수반법과 역전파.** 동역학 $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$와 비용 $J = \sum_t c(\mathbf{x}_t, \mathbf{u}_t)$에서 $\partial J / \partial \mathbf{u}_t$는 8.4장의 역전파(reverse-mode 자동미분)로 계산한다. 제어 이론에서는 같은 계산을 **수반**(adjoint) 방법이라 부른다. 수반 변수 $\boldsymbol{\lambda}_t = \partial J / \partial \mathbf{x}_t$를 마지막 시각에서 시작해 거꾸로 전파한다.

$$
\boldsymbol{\lambda}_T = \nabla_{\mathbf{x}} c_T, \qquad
\boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c_t + \Big(\frac{\partial f}{\partial \mathbf{x}_t}\Big)^{\!\top} \boldsymbol{\lambda}_{t+1}, \qquad
\frac{\partial J}{\partial \mathbf{u}_t} = \nabla_{\mathbf{u}} c_t + \Big(\frac{\partial f}{\partial \mathbf{u}_t}\Big)^{\!\top} \boldsymbol{\lambda}_{t+1}.
$$

정방향 시뮬레이션 한 번과 역방향 전파 한 번으로 모든 시각의 기울기를 얻는다. 정책 $\mathbf{u}_t = \pi_\theta(\mathbf{x}_t)$의 파라미터에 대한 기울기도 같은 방식으로 이어서 계산한다. 이것이 미분가능 시뮬레이션으로 정책을 최적화하는 방법의 뼈대다.

이중 적분기의 40 step 개루프 입력열을 수반법 기울기로 최적화하고, 불안정한 시스템에서 기울기가 지평에 따라 어떻게 커지는지 본다.

```python {exec="browser" session="control"}
def grad_open_loop(A_, U, x0_, Qm, r):       # J = sum_t x_t^T Q x_t + r u_t^2,  x_{t+1} = A x_t + B u_t
    T, xs = len(U), [x0_]
    for t in range(T):
        xs.append(A_ @ xs[-1] + B[:, 0] * U[t])
    lam, g = 2 * Qm @ xs[T], np.zeros(T)     # 수반(adjoint) 변수 lam_t = dJ/dx_t
    for t in reversed(range(T)):
        g[t] = 2 * r * U[t] + B[:, 0] @ lam  # dJ/du_t
        lam = 2 * Qm @ xs[t] + A_.T @ lam
    return g, sum(x @ Qm @ x for x in xs) + r * np.sum(U**2)

U = np.zeros(40)
g, J = grad_open_loop(A, U, x0, Qc, 0.1)
eps = 1e-6; e = np.zeros(40); e[7] = eps
num = (grad_open_loop(A, U + e, x0, Qc, 0.1)[1] - J) / eps
print(f"수반법 dJ/du_7 = {g[7]:.4f}, 유한차분 = {num:.4f}")
for _ in range(300):                         # 기울기 하강으로 개루프 입력열 최적화
    g, J = grad_open_loop(A, U, x0, Qc, 0.1)
    U -= 0.02 * g
Pt = Qc.copy()
for _ in range(40):                          # 같은 40 step 문제의 정확한 최적값(유한 지평 Riccati)
    Pt = Qc + A.T @ Pt @ A - A.T @ Pt @ B @ np.linalg.solve(Rc + B.T @ Pt @ B, B.T @ Pt @ A)
print(f"기울기 하강 300번 뒤 비용 {J:.4f}, 유한 지평 Riccati 최적값 {x0 @ Pt @ x0:.4f}")
A_unst = np.array([[1.0, dt_s], [dt_s * 9.81, 1.0]])   # 뒤집힌 진자의 선형화: 불안정
for T in (10, 40, 80):
    g, _ = grad_open_loop(A_unst, np.zeros(T), x0, Qc, 0.1)
    print(f"불안정 시스템, 지평 {T:2d}: |dJ/du_0| = {abs(g[0]):.2e}")
```

```output
수반법 dJ/du_7 = 10.8900, 유한차분 = 10.8900
기울기 하강 300번 뒤 비용 9.1069, 유한 지평 Riccati 최적값 9.0772
불안정 시스템, 지평 10: |dJ/du_0| = 1.53e+01
불안정 시스템, 지평 40: |dJ/du_0| = 1.94e+08
불안정 시스템, 지평 80: |dJ/du_0| = 5.69e+17
```

수반법의 기울기는 유한차분과 일치하고, 기울기 하강으로 정확한 최적값(유한 지평 Riccati)에 가까이 간다. 300번 뒤에도 0.03의 차이가 남는 것은 이 문제가 나쁜 조건(ill-conditioned)이라 1차 방법이 느리기 때문이다(10.3장). 더 중요한 것은 마지막 결과다. 뒤집힌 진자처럼 **불안정한** 시스템에서는 $\partial f/\partial \mathbf{x}$의 고윳값이 1보다 커서, 기울기가 지평의 길이에 대해 지수적으로 커진다. 지평 80 step에서 기울기가 $10^{17}$ 수준이다. 혼돈계(chaotic system)나 접촉·충돌처럼 동역학이 거의 불연속인 시뮬레이션에서도 같은 일이 생긴다.

**연구에서의 의미.** 미분가능 시뮬레이션은 동역학이 매끄럽고 지평이 적당할 때 매우 효율적이다. 기울기 폭발에 대한 대응은 순환 신경망의 경우(18.3장)와 비슷하다. 지평을 짧게 끊어 여러 구간으로 나누고, 기울기를 자르며, MPC처럼 짧은 지평의 최적화를 반복한다. 기울기가 정확하다고 최적화가 잘 되는 것은 아니라는 점, 그리고 시뮬레이터의 기울기가 실제 시스템의 기울기와 같다는 보장이 없다는 점(모형 오차)이 앞 절과 같은 한계다. 미분 가능한 ODE solver와 그 수반법은 24.3장에서 다시 다룬다.

## 실습

학습된 동역학 모형과 MPC로 제약이 있는 제어 문제를 풀고 실제 모형 대비 성능을 평가한다. 코드는 `labs/reinforcement-learning/model-based-control/mpc_learned_model.py`다. 감쇠 진자를 각도 0에서 0.8 rad로 옮겨 유지하는 문제이고, 토크 한계 $|u| \le 8$, 각속도 한계 $|\omega| \le 0.6$ rad/s가 있다. 세 가지 모형으로 같은 MPC(교차 엔트로피 방법, 앞 계획으로 warm start)를 돌리고, 실제 시스템에서 seed 5개로 평가한다.

- 실제 비선형 모형
- 데이터 500개로 적합한 선형 회귀 모형(특징 $\theta, \omega, u, 1$)
- 물리 특징 $\sin\theta$를 더한 회귀 모형

집필 시점의 실행 결과는 다음과 같다.

```output
모델                         비용 평균    마지막 20 step 각도 오차 [rad]      속도 제약 위반 step
실제 모델                     10.355                      0.053                0.0
선형 회귀 모델                  11.854                      0.110                0.0
sin 특징 회귀 모델              10.490                      0.052                0.0
```

세 가지를 읽는다.

- **세 모형 모두 속도 제약은 지켰다.** 제약은 MPC가 계획 안에서 다루므로 모형이 조금 틀려도 여유를 두고 지켜진다. 다만 모형 오차가 크면 예측상 지킨 제약을 실제로 어길 수 있으므로, 안전이 중요한 문제에서는 제약에 여유(margin)를 둔다.
- **선형 모형은 목표 각도 근처에서 오차가 두 배다.** 0.8 rad를 유지하려면 중력 토크 $g\sin(0.8)$을 정확히 상쇄해야 하는데, 선형 모형은 $\sin\theta$를 직선으로 근사해 그 토크를 잘못 예측한다. 1 step 예측 오차는 작았지만 정상 상태 오차로 나타난다.
- **물리 특징 하나가 차이를 없앤다.** $\sin\theta$를 특징에 넣은 회귀 모형은 실제 모형과 거의 같은 성능을 낸다. 모형의 구조가 맞으면 적은 데이터로도 계획에 쓸 만한 모형을 얻는다.

이어서 다음을 직접 해 본다.

1. 데이터 수를 50개로 줄이고, 데이터를 $|\theta| < 0.3$ 범위에서만 모아 같은 실험을 반복한다. 모형이 본 적 없는 영역(0.8 rad)으로 계획할 때 무슨 일이 생기는지 기록한다.
2. 같은 문제를 20.2장의 model-free 방법(상태를 격자로 나눈 Q-learning 등)으로 풀고, 실제 시스템과의 상호작용 step 수, 최종 성능, 제약 위반 횟수를 MPC와 표로 비교한다.

연습문제 두 개로 이 장의 두 계산을 직접 구현한다.

::exercise{id="lqr-riccati"}

::exercise{id="random-shooting-mpc"}

## 실패 사례

- **학습된 모델의 오차를 무시한 계획.** 1 step 오차로만 모형을 검증하고 긴 지평으로 계획하면, 계획은 모형의 오차를 "이용"하는 방향으로 간다. 최적화기는 모형이 실제보다 좋게 예측하는 행동을 찾아내기 때문이다. 모형을 여러 step 굴린 오차로 검증하고, 계획이 데이터가 덮지 않은 영역으로 가는지 확인한다.
- **입력을 잘라서 제약을 처리했다고 믿는 것.** 자른 LQR은 입력 한계만 지키고 상태 제약은 어겼다. 상태 제약은 계획 안에서(MPC) 또는 명시적인 안전 장치로 다뤄야 한다.
- **시뮬레이터에서의 성능을 실제 성능으로 보고하는 것.** 시뮬레이터 자체가 모형이다. 시뮬레이터에서 최적화한 정책은 시뮬레이터의 오차까지 이용한다. 실제 시스템(또는 다른 충실도의 시뮬레이터)에서의 검증을 따로 보고한다.
- **불안정한 시스템에서 긴 지평 기울기를 그대로 쓰는 것.** 지평 80 step에서 기울기는 $10^{17}$이었다. 학습이 발산하면 학습률보다 지평과 시스템의 안정성을 먼저 의심한다.

## 통과 기준

- 같은 제어 문제를 MPC와 RL로 풀어 장단점을 비교한다(필요한 상호작용 수, 최종 비용, 제약 위반 횟수를 표로 보고).
- 연습문제 `lqr-riccati`와 `random-shooting-mpc`의 테스트를 통과한다.
- 학습된 모형의 1 step 오차와 여러 step 굴린 오차를 함께 보고하고, 그 차이가 계획 성능에 미친 영향을 설명한다.

## 참고 문헌

- Sutton, Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — 모형을 이용한 계획과 학습, Dyna(8장)
- Chua, Calandra, McAllister, Levine (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. *NeurIPS 31*. — 확률적 동역학 모형의 ensemble과 샘플링 기반 MPC(PETS)
