---
id: scientific-discovery/equation-discovery
title: "방정식 발견과 기호 회귀"
title_en: "Equation Discovery and Symbolic Regression"
tier: advanced
status: draft
prerequisites: [scientific-ml/physics-and-learning, classical-ml/linear-models]
---

# 방정식 발견과 기호 회귀

신경망 대리모델은 정확하게 예측해도 "왜"를 말해 주지 않고, 학습 범위 밖에서는 근거 없이 행동한다. 과학자가 오래 써 온 형태의 지식은 다르다. $F = ma$나 $\dot{x} = ax - bxy$ 같은 **짧은 식**은 읽을 수 있고, 단위를 따져 볼 수 있고, 처음 보는 조건에 적용해 반증할 수 있다. 이 장은 데이터에서 그런 식을 찾는 두 가지 접근을 다룬다. 후보 항을 미리 정해 두고 그중 몇 개만 고르는 **희소 회귀**, 그리고 식의 구조 자체를 탐색하는 **기호 회귀**다.

식을 찾는 것보다 어려운 일은 찾은 식을 믿을 근거를 대는 일이다. 데이터에 잘 맞는 식은 무수히 많다. 그래서 이 장의 절반은 식의 복잡도를 다루는 법, 차원 분석으로 탐색 공간을 줄이는 법, 그리고 외삽·보존 법칙·물리적 극한·단위로 식을 검증하는 법에 쓴다.

## 학습 목표

- 데이터에서 지배 방정식을 찾는 방법을 이해한다
- 발견된 식의 타당성을 검증한다
- 차원 분석으로 식의 탐색 공간을 줄인다

## 희소 회귀로 동역학 찾기

**왜 필요한가.** 상태 $\mathbf{x}(t)$의 궤적을 관측했고, 그것이 미분방정식 $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$를 따른다고 믿는다. 24.6장의 시스템 식별은 $\mathbf{f}$의 **형태**를 알고 파라미터만 추정했다. 여기서는 형태도 모른다. 다만 많은 물리·생물·화학 시스템에서 $\mathbf{f}$는 몇 개의 단순한 항(선형 항, 두 변수의 곱, 낮은 차수의 다항식)의 합이다. 반응 속도식, 개체군 동역학, 대부분의 역학 방정식이 그렇다.

**정의.** SINDy(sparse identification of nonlinear dynamics, Brunton 외, 2016)는 이 관찰을 선형 회귀 문제로 바꾼다. 후보 함수 $p$개를 모은 **사전**(library) $\boldsymbol{\Theta}(\mathbf{X}) = [\,1,\ x,\ y,\ x^2,\ xy,\ y^2,\ \dots\,]$를 만들고, 시각 $T$개의 상태와 미분으로

$$
\dot{\mathbf{X}} \approx \boldsymbol{\Theta}(\mathbf{X})\,\boldsymbol{\Xi}, \qquad \dot{\mathbf{X}} \in \mathbb{R}^{T \times n},\ \boldsymbol{\Theta} \in \mathbb{R}^{T \times p},\ \boldsymbol{\Xi} \in \mathbb{R}^{p \times n}
$$

를 푼다. $\boldsymbol{\Xi}$의 열 $j$는 $\dot{x}_j$를 이루는 후보 항의 계수다. 핵심 가정은 $\boldsymbol{\Xi}$가 **희소**하다는 것, 즉 대부분의 계수가 0이라는 것이다. 희소 해를 찾는 방법으로 SINDy 논문은 **순차적 임계 최소제곱**(sequentially thresholded least squares, STLSQ)을 썼다. 최소제곱으로 풀고, 절댓값이 임계값 $\lambda$보다 작은 계수를 0으로 만들고, 남은 항만으로 다시 최소제곱을 푸는 것을 반복한다. 16.2장의 lasso($L_1$ 정규화)도 같은 목적으로 쓸 수 있다.

**직관.** 회귀가 "계수 몇 개로 데이터를 설명하라"는 제약을 받으면, 데이터를 만든 항이 가장 적은 수로 설명하는 조합이 된다. 잘못된 항으로도 데이터를 흉내 낼 수 있지만 대개 더 많은 항이 필요하다.

로트카–볼테라 포식자–먹이 모델로 확인한다. 먹이 $x$와 포식자 $y$는

$$
\dot{x} = a x - b x y, \qquad \dot{y} = c x y - d y
$$

를 따른다. $a$는 먹이의 증식률, $b$는 포식률, $c$는 포식이 포식자 증가로 이어지는 비율, $d$는 포식자의 사망률이다. 잡음 없는 궤적을 만들고, 미분은 관측값에서 유한차분으로 추정한다. 사전은 2차까지의 다항식 6개다.

```python {exec="browser" session="sindy"}
import numpy as np
from scipy.integrate import solve_ivp

a, b, c, d = 1.0, 0.5, 0.2, 0.6                       # 로트카–볼테라: 먹이 x, 포식자 y
lv = lambda t, s: [a * s[0] - b * s[0] * s[1], c * s[0] * s[1] - d * s[1]]
dt = 0.01
t = np.arange(0, 20, dt)                              # (T,)
X = solve_ivp(lv, (0, 20), [4.0, 2.0], t_eval=t, rtol=1e-10, atol=1e-10).y.T   # (T, 2) 상태 궤적

names = ["1", "x", "y", "x^2", "xy", "y^2"]
def library(X):                                       # (T, 2) -> (T, 6) 후보 함수 사전
    x, y = X[:, 0], X[:, 1]
    return np.stack([np.ones_like(x), x, y, x**2, x * y, y**2], axis=1)

def stlsq(Theta, dX, thr=0.05, iters=10):             # 순차적 임계 최소제곱
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]    # (6, 2)
    for _ in range(iters):
        small = np.abs(Xi) < thr
        Xi[small] = 0
        for j in range(dX.shape[1]):                  # 남은 항만으로 다시 최소제곱
            keep = ~small[:, j]
            Xi[keep, j] = np.linalg.lstsq(Theta[:, keep], dX[:, j], rcond=None)[0]
    return Xi

dX = np.gradient(X, dt, axis=0)                       # (T, 2) 유한차분 미분
Xi_clean = stlsq(library(X), dX)
for j, v in enumerate(["dx/dt", "dy/dt"]):
    print(v, "=", " ".join(f"{Xi_clean[i, j]:+.3f} {n}" for i, n in enumerate(names) if Xi_clean[i, j] != 0))
```

```output
dx/dt = +1.000 x -0.500 xy
dy/dt = -0.600 y +0.200 xy
```

6개 후보 중 정확히 맞는 네 항과 계수가 나왔다.

**미분 추정의 잡음 문제.** 실제 데이터에는 측정 잡음이 있고, SINDy의 약점은 미분이다. 관측값 $\tilde{x}_k = x_k + \eta_k$의 중앙 차분은

$$
\frac{\tilde{x}_{k+1} - \tilde{x}_{k-1}}{2\Delta t} = \dot{x}_k + O(\Delta t^2) + \frac{\eta_{k+1} - \eta_{k-1}}{2\Delta t}
$$

이다. 마지막 항의 표준편차는 $\sigma_\eta / (\sqrt{2}\,\Delta t)$이므로 $\Delta t = 0.01$이면 측정 잡음이 약 70배로 커진다. 샘플링을 촘촘하게 할수록 오히려 나빠진다. 표준적인 대응은 미분하기 전에 평활(smoothing)하는 것이다. 여기서는 창 안의 점들에 3차 다항식을 최소제곱으로 맞추고 그 다항식의 미분을 쓰는 Savitzky–Golay 필터(Savitzky, Golay, 1964)를 쓴다. 잡음 수준은 각 상태 변수의 표준편차에 대한 비율이다.

```python {exec="browser" session="sindy"}
from scipy.signal import savgol_filter

Xi_true = np.zeros((6, 2))
Xi_true[[1, 4], 0] = [a, -b]                          # dx/dt = a x - b xy
Xi_true[[2, 4], 1] = [-d, c]                          # dy/dt = -d y + c xy

def score(Xi):                                        # 잘못 들어간 항, 빠진 항, 계수 최대 오차
    extra = np.sum((Xi != 0) & (Xi_true == 0))
    missing = np.sum((Xi == 0) & (Xi_true != 0))
    return f"잘못 들어간 항 {extra:2d}  빠진 항 {missing}  계수 최대 오차 {np.abs(Xi - Xi_true).max():.3f}"

rng = np.random.default_rng(0)
found = {}                                            # 잡음 수준별로 찾은 계수
for level in [0.001, 0.01, 0.05, 0.2]:
    Xn = X + level * X.std(0) * rng.normal(size=X.shape)          # 측정 잡음
    dX_fd = np.gradient(Xn, dt, axis=0)                           # 유한차분: 잡음을 1/dt배로 키운다
    Xs = savgol_filter(Xn, 101, 3, axis=0)                        # 국소 다항식으로 평활
    dX_sg = savgol_filter(Xn, 101, 3, deriv=1, delta=dt, axis=0)  # 같은 다항식의 미분
    found[level] = stlsq(library(Xn), dX_fd), stlsq(library(Xs), dX_sg)
    print(f"잡음 {level:5.3f}  유한차분: {score(found[level][0])}")
    print(f"             평활 미분: {score(found[level][1])}")
```

```output
잡음 0.001  유한차분: 잘못 들어간 항  0  빠진 항 0  계수 최대 오차 0.000
             평활 미분: 잘못 들어간 항  0  빠진 항 0  계수 최대 오차 0.000
잡음 0.010  유한차분: 잘못 들어간 항  4  빠진 항 0  계수 최대 오차 2.494
             평활 미분: 잘못 들어간 항  0  빠진 항 0  계수 최대 오차 0.000
잡음 0.050  유한차분: 잘못 들어간 항  8  빠진 항 0  계수 최대 오차 10.474
             평활 미분: 잘못 들어간 항  0  빠진 항 0  계수 최대 오차 0.002
잡음 0.200  유한차분: 잘못 들어간 항  8  빠진 항 0  계수 최대 오차 6.096
             평활 미분: 잘못 들어간 항  8  빠진 항 0  계수 최대 오차 2.035
```

잡음 1%에서 이미 유한차분은 없는 항 4개를 찾아내고 계수도 크게 틀린다. 평활 미분은 잡음 5%까지 구조와 계수를 거의 정확히 복원한다. 그러나 잡음 20%에서는 평활해도 모든 후보 항이 들어간다. 평활은 잡음을 줄이는 대가로 신호도 무디게 하므로, 어떤 방법이든 복원이 무너지는 잡음 수준이 있다. 논문이나 보고서에서 방정식 발견 결과를 말할 때는 그 경계가 어디인지를 함께 보여야 한다.

::quiz{id="finite-difference-noise"}

희소 회귀에는 구조적인 한계도 있다. **사전에 없는 항은 찾을 수 없다.** 참 동역학에 $\sin x$나 $x/(1 + x)$(포화형 반응) 같은 항이 있는데 사전이 다항식뿐이면, STLSQ는 다항식 여러 개로 그 항을 흉내 낸 식을 돌려준다. 계수가 정확히 맞고 항이 적은 결과가 나오면 사전이 맞았다는 증거가 되지만, 항이 많고 계수가 지저분하면 사전이 틀렸을 가능성을 먼저 의심한다. 또 후보 항끼리 강하게 상관되면($x$와 $x^3$이 좁은 범위에서만 관측된 경우) 어느 항이 진짜인지 회귀가 가리지 못한다. 상태 공간을 넓게 덮는 궤적, 여러 초기 조건의 데이터가 필요하다.

:::note[분야 예시]
화학 반응 네트워크에서는 농도의 곱(질량 작용 법칙) 항으로 사전을 만들면 반응 경로 후보가 곧 사전의 열이 된다. 유체역학에서는 공간 미분($u_x$, $u_{xx}$, $u u_x$)을 사전에 넣어 편미분방정식을 찾는 변형이 쓰인다. 배터리 열 모델(30장)에서는 전류의 제곱(줄 열)과 온도 차(냉각) 항을 사전에 넣고 셀 온도 기록에서 열 수지 식을 확인할 수 있다. 어느 경우든 사전을 정하는 일이 곧 물리적 가설을 세우는 일이다.
:::

## 기호 회귀

**왜 필요한가.** 희소 회귀는 "어떤 항이 있는가"만 고른다. 식의 모양, 즉 $\sqrt{L/g}$처럼 제곱근 안에 비가 들어가는지, 지수 안에 무엇이 들어가는지는 사전을 만든 사람이 미리 정해야 한다. **기호 회귀**(symbolic regression)는 식의 구조까지 데이터에서 찾는다.

**정의.** 식을 연산자(+, ×, ÷, sin, exp, 거듭제곱)와 변수·상수로 이루어진 트리로 표현하고, 트리의 공간을 탐색해 데이터에 맞는 식을 찾는다. 이 공간은 트리 크기에 따라 기하급수적으로 커지므로 전수 탐색은 작은 식에서만 가능하다. 대표적인 탐색 방법은 다음과 같다.

- **유전 프로그래밍**(genetic programming): 식 트리의 집단을 두고, 부분 트리를 바꾸거나(돌연변이) 두 트리의 부분을 교환(교차)하며 오차와 복잡도가 좋은 식을 남긴다. Schmidt와 Lipson(2009)은 이 방식으로 단순 조화 진동자에서 혼돈적인 이중 진자까지 여러 역학계의 운동 측정 데이터에서 해밀토니안 같은 보존 법칙을 찾았다.
- **신경망과 결합한 방법**: 신경망으로 데이터를 먼저 매끄럽게 맞춘 뒤 그 신경망의 성질(대칭성, 변수 분리 가능성)을 조사해 문제를 작은 부분 문제로 쪼개는 방법이 있다. Udrescu와 Tegmark(2020)의 AI Feynman은 차원 분석, 다항식 맞춤, 대칭성·분리 가능성 검사를 재귀적으로 적용한다. 신경망이 식 토큰을 직접 생성하도록 학습하는 방법도 연구되고 있다.

어느 방법이든 탐색 공간이 너무 크다는 문제가 있다. 물리학이 오래 써 온 가장 강력한 탐색 공간 축소 도구가 **차원 분석**이다.

**Buckingham π 정리.** 물리 법칙은 단위의 선택에 의존하지 않아야 한다. 미터를 피트로 바꿨다고 진자의 주기가 달라지지는 않는다. 이 요구를 식으로 쓰면 다음과 같다. 관련된 물리량이 $n$개이고, 각 물리량의 차원을 기본 차원(질량 M, 길이 L, 시간 T 등) $k$개의 거듭제곱 지수로 쓴 **차원 행렬** $\mathbf{D} \in \mathbb{R}^{k \times n}$의 rank가 $r$이면, 법칙은 **무차원 조합 $n - r$개 사이의 관계**로 쓸 수 있다(Buckingham, 1914). 무차원 조합 $\Pi = \prod_i q_i^{e_i}$의 지수 벡터 $\mathbf{e}$는

$$
\mathbf{D}\,\mathbf{e} = \mathbf{0}
$$

을 만족하므로, π 그룹을 찾는 일은 차원 행렬의 영공간(7.3장)을 찾는 일이다.

진자의 주기 $T$가 줄 길이 $L$, 중력가속도 $g$, 추의 질량 $m$, 진폭 $\theta_0$에 의존한다고 하자. 변수 5개, 기본 차원 3개다. 반복 변수 $L, g, m$을 정하고, 나머지 변수 각각을 반복 변수의 거듭제곱으로 무차원화한다.

```python {exec="browser" session="sindy"}
from scipy.special import ellipk

dims = {"T": [0, 0, 1], "L": [0, 1, 0], "g": [0, 1, -2],   # [질량, 길이, 시간]의 지수
        "m": [1, 0, 0], "theta0": [0, 0, 0]}               # 주기 T, 줄 길이 L, 중력가속도 g, 질량 m, 진폭 theta0
D = np.array(list(dims.values())).T                         # (3, 5) 차원 행렬
print("변수 5개, 차원 행렬의 rank", np.linalg.matrix_rank(D), "-> π 그룹", 5 - np.linalg.matrix_rank(D), "개")

rep = ["L", "g", "m"]                                       # 반복 변수: 기본 차원 셋을 모두 담는다
D_rep = np.array([dims[v] for v in rep]).T                  # (3, 3)
for q in ["T", "theta0"]:
    e = np.linalg.solve(D_rep, -np.array(dims[q])) + 0.0    # q * L^e0 * g^e1 * m^e2 가 무차원
    print(f"Π = {q} · " + " · ".join(f"{v}^({x:+.1f})" for v, x in zip(rep, e)))

def period_s(L_m, g_mps2, m_kg, theta0):                    # 진자 주기의 정확한 식(타원 적분)
    return 4 * np.sqrt(L_m / g_mps2) * ellipk(np.sin(theta0 / 2) ** 2)

rng = np.random.default_rng(0)
L_m, g_mps2, m_kg = rng.uniform(0.1, 2, 100), rng.uniform(1.6, 25, 100), rng.uniform(0.01, 5, 100)
for theta0 in [0.2, 1.0]:
    T_s = period_s(L_m, g_mps2, m_kg, theta0)               # (100,) 길이·중력·질량이 제각각
    Pi1 = T_s * np.sqrt(g_mps2 / L_m)                       # (100,)
    print(f"theta0 = {theta0}: T 범위 {T_s.min():.2f}–{T_s.max():.2f} s,  Π1 = T·sqrt(g/L) 범위 {Pi1.min():.4f}–{Pi1.max():.4f}")
```

```output
변수 5개, 차원 행렬의 rank 3 -> π 그룹 2 개
Π = T · L^(-0.5) · g^(+0.5) · m^(+0.0)
Π = theta0 · L^(+0.0) · g^(+0.0) · m^(+0.0)
theta0 = 0.2: T 범위 0.47–5.88 s,  Π1 = T·sqrt(g/L) 범위 6.2989–6.2989
theta0 = 1.0: T 범위 0.50–6.25 s,  Π1 = T·sqrt(g/L) 범위 6.7000–6.7000
```

π 그룹은 $\Pi_1 = T\sqrt{g/L}$와 $\Pi_2 = \theta_0$ 두 개이므로, 법칙은 반드시 $T\sqrt{g/L} = \phi(\theta_0)$ 꼴이다. 변수 4개짜리 함수를 찾는 문제가 변수 1개짜리 함수를 찾는 문제로 줄었다. 질량 $m$은 질량 차원을 가진 유일한 변수라 어떤 무차원 조합에도 들어갈 수 없다. 데이터를 보기 전에 "주기는 질량과 무관하다"는 결론이 나온 것이다. 길이·중력·질량이 제각각인 진자 100개의 주기는 0.5초에서 6초까지 퍼지지만, $\Pi_1$로 바꾸면 진폭마다 한 값으로 모인다. 작은 진폭에서는 그 값이 $2\pi \approx 6.283$에 가깝다.

::quiz{id="pi-group-count"}

$\phi$의 모양은 차원 분석이 알려 주지 않는다. 이제 이것을 기호 회귀로 찾는다. 진폭 0.05–1.5 rad의 측정 40개(0.1% 잡음)에서, 후보 함수 7개 중 1–3개의 선형 결합을 **모두** 나열해 맞춘다. 실제 기호 회귀가 유전 프로그래밍으로 훑는 공간을 아주 작게 줄여 전수 탐색한 것이다. 각 식의 **복잡도**는 식 트리의 노드 수에 계수 수를 더해 센다. 그리고 학습 구간 밖인 1.6–2.4 rad에서의 오차(외삽 RMSE)를 함께 잰다.

```python {exec="browser" session="sindy"}
from itertools import combinations
rng = np.random.default_rng(1)
th_tr = np.sort(rng.uniform(0.05, 1.5, 40))                 # (40,) 학습 구간 진폭 [rad]
y_tr = 4 * ellipk(np.sin(th_tr / 2) ** 2) * (1 + 0.001 * rng.normal(size=40))   # Π1, 0.1% 측정 잡음
th_te = np.linspace(1.6, 2.4, 9)                            # 외삽 구간
y_te = 4 * ellipk(np.sin(th_te / 2) ** 2)

basis = {"1": (np.ones_like, 1), "θ": (lambda t: t, 1), "θ^2": (lambda t: t**2, 3),
         "θ^4": (lambda t: t**4, 3), "sin θ": (np.sin, 2), "cos θ": (np.cos, 2),
         "exp θ": (np.exp, 2)}                              # 이름: (함수, 식 트리의 노드 수)
def design(terms, th):                                      # (len(th), len(terms)) 설계 행렬
    return np.stack([f(th) for f, _ in (basis[n] for n in terms)], 1)

def pareto(names):                                          # 항 1–3개짜리 선형 결합을 모두 나열
    rows = []
    for k in (1, 2, 3):
        for terms in combinations(names, k):
            A = design(terms, th_tr)
            w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
            ext = design(terms, th_te) @ w - y_te
            cx = sum(basis[n][1] + 1 for n in terms)        # 복잡도: 노드 수 + 계수
            rows.append((cx, np.sqrt(np.mean((A @ w - y_tr) ** 2)), np.sqrt(np.mean(ext**2)),
                         " + ".join(f"{v:.3f}·{n}" for v, n in zip(w, terms))))
    best = np.inf
    for cx, rmse, ext, f in sorted(rows):                   # 복잡도 순으로, 오차가 5% 넘게 줄 때만 남긴다
        if rmse < 0.95 * best:
            best = rmse
            print(f"  복잡도 {cx:2d}  학습 RMSE {rmse:.4f}  외삽 RMSE {ext:.3f}  {f}")

print("모든 후보 함수"); pareto(list(basis))
print("짝함수만 (θ → -θ 대칭)"); pareto(["1", "θ^2", "θ^4", "cos θ"])
```

```output
모든 후보 함수
  복잡도  2  학습 RMSE 0.3062  외삽 RMSE 1.999  6.614·1
  복잡도  4  학습 RMSE 0.0832  외삽 RMSE 1.133  6.059·1 + 0.714·θ
  복잡도  5  학습 RMSE 0.0259  외삽 RMSE 0.278  5.886·1 + 0.307·exp θ
  복잡도  6  학습 RMSE 0.0120  외삽 RMSE 0.480  6.267·1 + 0.447·θ^2
  복잡도  7  학습 RMSE 0.0059  외삽 RMSE 0.069  5.825·1 + -0.320·θ + 0.438·exp θ
  복잡도  9  학습 RMSE 0.0049  외삽 RMSE 0.117  7.066·1 + 0.057·θ^4 + -0.784·cos θ
짝함수만 (θ → -θ 대칭)
  복잡도  2  학습 RMSE 0.3062  외삽 RMSE 1.999  6.614·1
  복잡도  5  학습 RMSE 0.0273  외삽 RMSE 0.891  7.309·1 + -1.063·cos θ
  복잡도  6  학습 RMSE 0.0120  외삽 RMSE 0.480  6.267·1 + 0.447·θ^2
  복잡도  9  학습 RMSE 0.0049  외삽 RMSE 0.117  7.066·1 + 0.057·θ^4 + -0.784·cos θ
```

## 해석 가능성과 단순성

**정확도와 복잡도의 trade-off.** 위 출력은 복잡도 순으로 훑으며 학습 오차가 이전보다 5% 넘게 줄어든 식만 남긴 것이다. 이렇게 남은 식들이 **Pareto front**다. front 위의 어떤 식도 다른 식보다 "복잡도와 오차 둘 다 나은" 경우가 없다. 더 단순하려면 오차를 감수해야 하고, 오차를 줄이려면 복잡도를 감수해야 한다. 기호 회귀의 산출물은 식 하나가 아니라 이 front이고, 그중 무엇을 고를지는 별도의 판단이다.

흔히 쓰는 기준은 **무릎**(knee)이다. 복잡도를 조금 늘려 오차가 크게 줄어드는 구간이 끝나는 곳이다. 이 예에서는 학습 RMSE가 복잡도 9에서 0.0049가 된 뒤로는 5% 넘게 줄어드는 식이 없다. 측정 잡음(0.1%, 약 0.006)과 같은 수준이다. 잡음보다 작은 학습 오차는 잡음을 외우고 있다는 뜻이므로, 그보다 복잡한 식을 고를 이유가 없다.

front를 읽는 데 두 가지를 더 확인한다.

- **모든 후보 함수**를 쓴 front에는 $\exp\theta$와 $\theta$가 들어간 식이 있다. 복잡도 7의 $5.825 - 0.320\theta + 0.438 e^{\theta}$는 학습 오차가 작고 외삽 오차(0.069)는 가장 작다. 그러나 진자의 주기는 진폭의 부호와 무관해야 한다. 왼쪽으로 1 rad 당기든 오른쪽으로 당기든 같다. 이 식은 그 대칭을 어긴다. 학습 데이터가 양의 진폭뿐이어서 회귀는 이를 알 수 없었다.
- **짝함수만** 허용하면, 즉 $\theta \to -\theta$ 대칭을 탐색 공간에 넣으면 front는 상수, $7.309 - 1.063\cos\theta$, $6.267 + 0.447\theta^2$, $7.066 + 0.057\theta^4 - 0.784\cos\theta$로 이어진다. 마지막 식을 $\cos\theta \approx 1 - \theta^2/2 + \theta^4/24$로 전개하면 약 $6.282 + 0.392\theta^2 + 0.024\theta^4$이다. 큰 진폭 진자의 주기를 급수로 전개한 결과 $2\pi\,(1 + \theta_0^2/16 + 11\theta_0^4/3072 + \cdots) \approx 6.283 + 0.393\theta_0^2 + 0.022\theta_0^4$와 거의 같다. 기호 회귀가 "참 식"인 타원 적분을 찾지는 못했지만, 학습 구간에서 그 식의 저차 근사를 찾은 것이다.

::quiz{id="pareto-knee"}

**해석 가능성의 한계.** 짧은 식이 읽기 쉽다는 것과 그 식이 참이라는 것은 다른 문제다. 위의 $\exp\theta$ 식도 짧고 읽을 수 있다. 해석 가능한 식의 장점은 **틀렸을 때 틀린 것을 확인할 수 있다**는 데 있다. 대칭을 넣어 보고, 극한을 계산해 보고, 단위를 따져 볼 수 있다. 신경망 대리모델에서는 이런 점검이 어렵다. 다음 절이 그 점검이다.

## 검증

학습 구간에서 잘 맞는다는 것은 발견한 식을 법칙으로 받아들이기 위한 출발점일 뿐이다. 다음 네 가지를 차례로 확인한다.

1. **새 조건에서의 예측.** 학습에 쓰지 않은 초기 조건, 매개변수, 구간에서 예측한다. 동역학 모델이면 찾은 식을 적분해 궤적 전체를 비교한다. 미분값 하나하나가 맞아도 오차가 쌓이면 궤적은 벗어날 수 있다.
2. **보존 법칙과 구조.** 참 시스템에 보존량이나 대칭이 있으면 찾은 식도 그것을 지켜야 한다. 로트카–볼테라 모델은 $H = c x - d \ln x + b y - a \ln y$를 보존한다(시간 미분이 0이 됨을 대입해 확인할 수 있다).
3. **물리적 극한.** 알려진 극한에서 알려진 값이 나와야 한다. 진자의 진폭이 0으로 가면 주기는 단진자 공식 $2\pi\sqrt{L/g}$, 즉 $\Pi_1 \to 2\pi$가 되어야 한다.
4. **단위 일관성.** 식의 모든 항이 같은 단위를 가져야 하고, 초월함수(exp, log, sin)의 인자는 무차원이어야 한다. π 그룹으로 쓴 식은 이 조건을 자동으로 만족한다. 차원이 있는 변수로 직접 기호 회귀를 하면 $e^{L}$이나 $L + g$ 같은 식이 front에 나올 수 있고, 이런 식은 계수가 아무리 잘 맞아도 받아들이지 않는다.

```python {exec="browser" session="sindy"}
def H(X):                                                   # 로트카–볼테라의 보존량
    return c * X[:, 0] - d * np.log(X[:, 0]) + b * X[:, 1] - a * np.log(X[:, 1])

s_new = [1.0, 1.0]                                          # 학습 궤적에 없던 초기 조건
X_new = solve_ivp(lv, (0, 20), s_new, t_eval=t, rtol=1e-10, atol=1e-10).y.T
print("참 모델의 보존량 변동폭 < 1e-6:", np.ptp(H(X_new)) < 1e-6)
for name, Xi in [("잡음 0.05, 평활 미분", found[0.05][1]), ("잡음 0.2,  평활 미분", found[0.2][1])]:
    rhs = lambda _, s: library(np.array([s]))[0] @ Xi       # 찾은 식을 그대로 적분
    Xp = solve_ivp(rhs, (0, 20), s_new, t_eval=t, rtol=1e-8).y.T
    ok = len(Xp) == len(t) and np.all(Xp > 0)
    if ok:
        print(f"{name}: 새 초기 조건 RMSE {np.sqrt(np.mean((Xp - X_new) ** 2)):.3f}  보존량 변동폭 {np.ptp(H(Xp)):.3f}")
    else:
        print(f"{name}: 적분 실패 — 개체 수가 음수가 되거나 발산한다")

def fit_predict(terms, th):                                 # 셀 4와 같은 최소제곱으로 맞춘 뒤 th에서 예측
    w = np.linalg.lstsq(design(terms, th_tr), y_tr, rcond=None)[0]
    return design(terms, np.atleast_1d(th)) @ w

for terms in [("1", "θ^2"), ("1", "θ", "exp θ"), ("1", "θ^4", "cos θ")]:
    f0 = fit_predict(terms, 0.0)[0]
    asym = abs(fit_predict(terms, 1.0)[0] - fit_predict(terms, -1.0)[0])
    print(f"{' + '.join(terms):16s} θ→0 극한 {f0:.3f} (정답 2π = {2 * np.pi:.3f})   |f(1) - f(-1)| = {asym:.3f}")
```

```output
참 모델의 보존량 변동폭 < 1e-6: True
잡음 0.05, 평활 미분: 새 초기 조건 RMSE 0.022  보존량 변동폭 0.002
잡음 0.2,  평활 미분: 적분 실패 — 개체 수가 음수가 되거나 발산한다
1 + θ^2          θ→0 극한 6.267 (정답 2π = 6.283)   |f(1) - f(-1)| = 0.000
1 + θ + exp θ    θ→0 극한 6.263 (정답 2π = 6.283)   |f(1) - f(-1)| = 0.389
1 + θ^4 + cos θ  θ→0 극한 6.282 (정답 2π = 6.283)   |f(1) - f(-1)| = 0.000
```

잡음 5%에서 평활 미분으로 찾은 로트카–볼테라 식은 처음 보는 초기 조건에서도 궤적 RMSE 0.022로 참 궤적을 따라가고, 보존량의 변동폭도 0.002로 작다. 잡음 20%에서 찾은, 모든 후보 항이 들어간 식은 적분 자체가 실패한다. 학습 데이터에서의 회귀 오차만으로는 이 차이가 드러나지 않는다. 진자 후보 식에서는 $\exp\theta$ 식이 대칭 점검에서 $|f(1) - f(-1)| = 0.389$로 탈락하고, 짝함수 식 둘 중 $\theta^4 + \cos\theta$ 식이 작은 진폭 극한 $2\pi$를 소수 셋째 자리까지 맞힌다. $1 + \theta^2$ 식은 대칭은 지키지만 극한에서 0.016 벗어난다. 학습 구간의 큰 진폭 데이터에 맞추느라 상수항이 밀렸기 때문이다.

:::note[분야 예시]
천문학의 케플러 제3법칙(공전 주기의 제곱이 궤도 긴반지름의 세제곱에 비례)은 행성 관측표에서 찾은 거듭제곱 법칙의 고전적 예이고, 이후 뉴턴 역학에서 유도되며 검증되었다. 재료과학에서는 결정립 크기와 항복 강도의 관계 같은 경험식을 데이터에서 찾고, 그 식이 다른 합금·온도에서도 성립하는지로 적용 범위를 정한다. 기후 모델의 아격자 매개화(27.3장)에서는 고해상도 시뮬레이션에서 찾은 식을 저해상도 모델에 넣었을 때 장기 적분이 안정한지가 결정적인 검증이다. 학습 데이터에서 잘 맞는 매개화도 결합 모델 안에서 불안정해질 수 있다.
:::

## 실습

본문의 셀 다섯 개는 `labs/scientific-discovery/equation-discovery/cells.py`에 있다. 알려진 동역학 시스템의 잡음 섞인 궤적에서 지배 방정식을 복원하고, 잡음 수준에 따른 실패 경계를 확인한다.

1. 셀 2의 잡음 수준을 0.01에서 0.2까지 촘촘히 바꾸며, 평활 미분 + STLSQ가 정확한 구조를 복원하는 마지막 잡음 수준을 찾는다. Savitzky–Golay 창 크기(51, 101, 201)와 STLSQ 임계값(0.02, 0.05, 0.1)에 따라 그 경계가 어떻게 움직이는지 표로 정리한다.
2. 사전에 3차 항($x^3$, $x^2 y$, $x y^2$, $y^3$)을 추가하고 같은 실험을 반복한다. 후보가 많아지면 잘못 들어간 항이 늘어나는지 확인한다.
3. 참 동역학의 포식 항을 포화형 $b x y / (1 + x)$로 바꾸고, 다항식 사전으로 무엇이 나오는지 본다. 사전에 없는 항을 다항식이 어떻게 흉내 내는지, 그 식이 새 초기 조건에서 어떻게 실패하는지 기록한다.
4. 찾은 식마다 셀 5의 네 가지 검증(새 조건 궤적, 보존량, 극한, 단위)을 적용해 결과를 한 표로 보고한다.

연습문제 두 개에서 STLSQ와 π 그룹 계산을 직접 구현한다.

::exercise{id="stlsq"}

::exercise{id="pi-groups"}

## 실패 사례

- **학습 구간에 잘 맞는 복잡한 식을 법칙으로 주장하는 것.** 학습 오차가 측정 잡음보다 작아졌다면 잡음을 외우고 있다는 신호다. Pareto front의 무릎에서 고르고, 학습 구간 밖과 새 조건에서 검증한 뒤에만 법칙이라고 부른다.
- **단위가 맞지 않는 식을 받아들이는 것.** $e^{L}$이나 $T + L$처럼 차원이 맞지 않는 식은 단위를 바꾸면 결과가 바뀐다. 기호 회귀는 π 그룹으로 만든 무차원 변수로 하거나, 탐색 중에 단위 일관성을 강제한다.
- **유한차분 미분을 그대로 쓰는 것.** 잡음이 $1/\Delta t$배로 커져 잡음 1%에서도 엉뚱한 항이 들어간다. 평활 미분, 적분형(약형) 정식화 등으로 미분 추정을 따로 검증한다.
- **사전에 없는 항을 찾을 수 있다고 믿는 것.** 희소 회귀는 주어진 후보 중에서만 고른다. 계수가 지저분하고 항이 많으면 사전이 틀렸을 가능성을 먼저 의심한다.
- **대칭과 극한을 확인하지 않는 것.** 학습 데이터가 한쪽 범위만 덮으면 대칭을 어기는 식도 잘 맞는다. 알려진 대칭과 극한은 탐색 공간에 넣거나 사후 점검으로 반드시 확인한다.

## 통과 기준

- 알려진 동역학 시스템의 잡음 섞인 궤적에서 지배 방정식을 복원하고, 복원이 무너지는 잡음 수준을 미분 추정 방법별로 보고한다.
- 발견한 식을 새 조건에서의 예측(외삽), 보존 법칙, 물리적 극한, 단위 일관성으로 검증한 보고서를 쓴다.
- 연습문제 두 개(`stlsq`, `pi-groups`)의 테스트를 통과한다.

## 참고 문헌

- Brunton, Proctor, Kutz (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932–3937. — SINDy: 후보 함수 사전과 순차적 임계 최소제곱
- Schmidt, Lipson (2009). Distilling free-form natural laws from experimental data. *Science*, 324(5923), 81–85. — 유전 프로그래밍 기반 기호 회귀로 역학계의 보존량 발견
- Udrescu, Tegmark (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6, eaay2631. — 차원 분석·대칭성·분리 가능성을 결합한 기호 회귀
- Buckingham (1914). On physically similar systems; illustrations of the use of dimensional equations. *Physical Review*, 4, 345–376. — π 정리
- Savitzky, Golay (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36, 1627–1639. — 국소 다항식 평활과 미분
