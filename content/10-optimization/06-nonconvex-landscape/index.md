---
id: optimization/nonconvex-landscape
title: "비볼록 최적화와 손실 지형"
title_en: "Nonconvex Optimization and Loss Landscapes"
tier: advanced
status: draft
prerequisites: [optimization/adaptive-methods]
---

# 비볼록 최적화와 손실 지형

10.3장의 수렴 이론은 대부분 볼록 함수에 대한 것이었고, 비볼록 함수에 대해서는 "gradient가 작은 점을 언젠가 지난다"는 약한 보장만 있었다. 그런데 신경망의 손실은 볼록이 아니다. 10.1장에서 본 것처럼 은닉 뉴런 하나짜리 신경망도 볼록이 아니고, 뉴런의 순서를 바꾸는 대칭만으로도 같은 손실을 갖는 해가 수없이 많다. 그럼에도 경사하강 계열의 방법은 실제로 잘 작동한다. 이 장은 그 사이의 간극을 다룬다. 비볼록 손실 지형에서 무엇이 실제로 학습을 막는지, 어떤 해가 선택되는지, 학습률 같은 설정이 어떤 해를 고르는지 본다.

이 주제에는 확립된 결과와 진행 중인 논쟁이 섞여 있다. 안장점이 정체를 만든다는 것, 학습률이 머물 수 있는 최솟점의 곡률에 상한을 건다는 것은 계산으로 확인할 수 있다. 반면 "평평한 최솟점이 일반화가 좋다"는 주장은 반례와 단서가 많다. 이 장은 둘을 구분하고, 손실 지형을 직접 측정하는 도구(선형 보간, 단면, Hessian 스펙트럼)를 쓰는 법과 그 결과를 과하게 해석하지 않는 법을 다룬다.

## 학습 목표

- 딥러닝 손실 지형의 특성을 이해한다
- 손실 지형 분석 도구를 사용한다

## 비볼록성

**국소해와 안장점.** 10.1장에서 stationary point를 Hessian의 고윳값으로 분류했다. 모든 고윳값이 양수이면 국소 최소, 부호가 섞이면 안장점이다. 저차원 직관으로는 비볼록 최적화의 위험이 "나쁜 국소 최솟점에 갇히는 것"으로 보인다. 고차원에서는 그림이 다르다.

**고차원에서의 안장점.** 직관적인 논증은 이렇다. stationary point의 Hessian을 무작위 대칭 행렬처럼 생각하면, 국소 최소가 되려면 $n$개의 고윳값이 **모두** 양수여야 한다. $n$이 커지면 그럴 확률은 빠르게 0으로 간다. 성분이 독립 정규분포인 무작위 대칭 행렬로 이 확률을 잰다.

```python {exec="browser" session="landscape"}
import numpy as np

rng = np.random.default_rng(0)
for n in [1, 2, 4, 6, 8]:
    A = rng.normal(size=(20000, n, n))
    S = (A + A.transpose(0, 2, 1)) / 2                    # (20000, n, n) 무작위 대칭 행렬
    pd = np.mean(np.linalg.eigvalsh(S)[:, 0] > 0)         # 최소 고윳값 > 0
    print(f"n = {n}  모든 고윳값이 양수일 비율 = {pd:.4f}")

def grad(p):                                              # f = x^2/2 - y^2/2 + y^4/4, 원점은 안장점
    return np.array([p[0], -p[1] + p[1] ** 3])
for eps in [1e-1, 1e-3, 1e-6, 1e-9, 0.0]:
    p, k, g_min = np.array([1.0, eps]), 0, np.inf
    while abs(p[1]) < 0.5 and k < 1000:                    # y가 0.5를 넘으면 안장점을 벗어난 것
        p, k = p - 0.1 * grad(p), k + 1
        g_min = min(g_min, np.linalg.norm(grad(p)))
    print(f"y0 = {eps:7.0e}  안장점을 벗어나기까지 {k:4d} step  (도중 최소 |grad| = {g_min:.1e})")
```

```output
n = 1  모든 고윳값이 양수일 비율 = 0.4980
n = 2  모든 고윳값이 양수일 비율 = 0.1450
n = 4  모든 고윳값이 양수일 비율 = 0.0028
n = 6  모든 고윳값이 양수일 비율 = 0.0000
n = 8  모든 고윳값이 양수일 비율 = 0.0000
y0 =   1e-01  안장점을 벗어나기까지   19 step  (도중 최소 |grad| = 3.9e-01)
y0 =   1e-03  안장점을 벗어나기까지   67 step  (도중 최소 |grad| = 3.8e-02)
y0 =   1e-06  안장점을 벗어나기까지  139 step  (도중 최소 |grad| = 1.0e-03)
y0 =   1e-09  안장점을 벗어나기까지  212 step  (도중 최소 |grad| = 2.7e-05)
y0 =   0e+00  안장점을 벗어나기까지 1000 step  (도중 최소 |grad| = 1.7e-46)
```

$n = 1$에서는 절반이 양수지만 $n = 4$에서 0.3%, $n = 6$부터는 2만 개 중 하나도 없다. 고윳값이 서로 독립이라면 $(1/2)^n$이었겠지만, 무작위 행렬의 고윳값은 서로 밀어내기 때문에 그보다도 빨리 줄어든다. Dauphin 외(2014)는 이런 무작위 행렬 이론의 결과와 신경망 실험을 근거로, 고차원 비볼록 최적화의 주된 어려움은 국소 최솟점보다 안장점이며, 손실이 높은 stationary point일수록 음의 고윳값이 많은 안장점일 가능성이 크다고 주장했다. 실제 손실의 Hessian은 무작위 행렬이 아니므로 이것은 정량적 법칙이 아니라 방향을 알려 주는 모형으로 받아들인다.

**안장점이 만드는 정체.** 안장점은 경사하강을 영원히 붙잡지는 않는다. 음의 곡률 $\lambda < 0$ 방향의 성분은 매 step $(1 - \eta\lambda) > 1$배가 되어 결국 커진다. 문제는 속도다. 셀의 두 번째 부분은 $f = x^2/2 - y^2/2 + y^4/4$의 안장점(원점)을 지나는 경사하강이다. 불안정한 방향의 처음 성분 $y_0$이 $10^{-3}$에서 $10^{-9}$로 줄면 벗어나는 데 걸리는 step이 67에서 212로 늘고, 그동안 gradient 노름은 $2.7 \times 10^{-5}$까지 내려간다. 성분이 매 step 1.1배씩 커지므로 정체 시간은 $\log(1/y_0)$에 비례한다. $y_0 = 0$이면 안장점의 안정 다양체 위에 있어 영원히 빠져나오지 못한다. 실제 학습에서는 mini-batch 잡음과 부동소수점 오차가 이런 정확한 정렬을 깨뜨린다.

손실 곡선에서 이 현상은 **평평한 구간 뒤의 급격한 하락**으로 나타난다. gradient가 작다는 이유로 평평한 구간을 수렴으로 판단하면 학습을 너무 일찍 멈추게 된다.

::quiz{id="saddle-plateau"}

**초기화 의존성.** 비볼록 함수에서는 시작점이 도착점을 정한다. 은닉 뉴런 8개짜리 tanh MLP로 $\sin(3x)$를 맞추며, 시드 20개로 초기화하고 같은 학습률로 full-batch 경사하강을 3000 step 한다.

```python {exec="browser" session="landscape"}
xs = np.linspace(-2, 2, 64)                                 # (N,) 입력
ys = np.sin(3 * xs)                                         # (N,) 목표
H = 8                                                       # 은닉 뉴런 수, theta = (w1, b1, w2, b2): 3H + 1개

def loss_grad(th):
    w1, b1, w2, b2 = th[:H], th[H:2 * H], th[2 * H:3 * H], th[3 * H]
    a = np.tanh(np.outer(xs, w1) + b1)                      # (N, H) 은닉 활성
    r = a @ w2 + b2 - ys                                    # (N,) 잔차
    da = np.outer(r, w2) * (1 - a**2) / len(xs)             # (N, H)
    g = np.concatenate([xs @ da, da.sum(0), a.T @ r / len(xs), [r.mean()]])
    return 0.5 * np.mean(r**2), g

def train(th, lr=0.2, steps=3000):
    for _ in range(steps):
        th = th - lr * loss_grad(th)[1]
    return th

sols = [train(np.random.default_rng(s).normal(size=3 * H + 1)) for s in range(20)]
finals = np.array([loss_grad(t)[0] for t in sols])
print(f"무작위 초기화 시드 20개: 최종 손실 최소 {finals.min():.5f}  중앙값 {np.median(finals):.5f}  최대 {finals.max():.5f}")
th_c = train(np.full(3 * H + 1, 0.5))                       # 모든 파라미터를 같은 값으로 초기화
print(f"모든 파라미터를 0.5로 초기화: 최종 손실 {loss_grad(th_c)[0]:.5f}, 서로 다른 w1 값 {np.unique(th_c[:H].round(8)).size}개")
```

```output
무작위 초기화 시드 20개: 최종 손실 최소 0.00020  중앙값 0.00438  최대 0.01009
모든 파라미터를 0.5로 초기화: 최종 손실 0.21642, 서로 다른 w1 값 1개
```

데이터·모델·학습률이 같아도 시드에 따라 최종 손실이 약 50배 차이 난다. 어떤 시드는 아직 정체 구간을 지나는 중이고, 어떤 시드는 다른 골짜기로 갔다. 모든 파라미터를 같은 값으로 초기화하면 더 극단적이다. 은닉 뉴런 8개가 처음부터 같은 값이므로 받는 gradient도 같고, 학습이 끝날 때까지 서로 다른 $w_1$ 값이 하나뿐이다. 뉴런 8개가 사실상 1개로 작동해 손실이 0.216에 머문다. 무작위 초기화의 첫째 역할은 이 **대칭을 깨는 것**이고, 초기 분산의 크기는 층을 지나며 신호와 gradient가 폭발하거나 사라지지 않게 정한다(17.1장).

:::note[분야 예시]
단백질 구조 예측이나 분자 구조 최적화처럼 에너지 지형이 험한 문제에서는 같은 계산을 여러 시작점에서 반복하는 것이 기본 절차다. 계산화학에서 여러 conformer에서 구조 최적화를 돌려 가장 낮은 에너지를 고르는 것, 신경망 연구에서 결과를 여러 시드의 평균과 편차로 보고하는 것은 같은 이유에서다. 한 번의 실행이 보여 주는 것은 지형의 한 골짜기뿐이다.
:::

## 평탄함과 일반화

**왜 필요한가.** 비볼록 손실에는 학습 데이터에서 손실이 거의 0인 해가 여러 개 있다. 그중 어떤 해가 시험 데이터에서 잘 동작하는지는 학습 손실만으로 알 수 없다. 오래된 가설 하나는 최솟점의 **모양**, 즉 주변이 평평한지 날카로운지가 일반화와 관련된다는 것이다.

**sharp · flat minima 논쟁.** 최솟점 근처의 곡률, 흔히 Hessian 최대 고윳값을 **sharpness**라 한다. 직관은 이렇다. 학습 데이터와 시험 데이터의 손실 지형은 조금 어긋나 있다. 평평한 최솟점은 지형이 조금 이동해도 손실이 크게 변하지 않지만, 날카로운 최솟점은 조금만 어긋나도 손실이 크게 오른다. Keskar 외(2017)는 큰 batch로 학습한 신경망이 Hessian의 큰 양의 고윳값이 많은 날카로운 최솟점으로 가고 일반화가 나쁘며, 작은 batch는 gradient 잡음 때문에 평평한 최솟점으로 간다는 실험 결과를 보고했다.

반론도 분명하다. Dinh 외(2017)는 ReLU 신경망의 대칭을 이용하면 **같은 함수**를 표현하면서 sharpness가 임의로 큰 파라미터를 만들 수 있음을 보였다. 함수가 같으면 시험 성능도 같으므로, 매개화에 따라 바뀌는 sharpness는 그 자체로 일반화의 척도가 될 수 없다. 가장 단순한 경우로 확인한다.

```python {exec="browser" session="landscape"}
xr = np.linspace(-1, 1, 41)
yr = 2.0 * np.maximum(0, xr)                                 # w1 * w2 = 2이면 정확히 맞는다
def L2(w):                                                   # 모델 w2 * relu(w1 * x)
    return 0.5 * np.mean((w[1] * np.maximum(0, w[0] * xr) - yr) ** 2)
def num_hess(fun, w, h=1e-4):
    E = np.eye(len(w)) * h
    return np.array([[(fun(w + E[i] + E[j]) - fun(w + E[i] - E[j]) - fun(w - E[i] + E[j]) + fun(w - E[i] - E[j]))
                      / (4 * h * h) for j in range(len(w))] for i in range(len(w))])
for a in [0.1, 1.0, np.sqrt(2), 10.0]:
    w = np.array([a, 2 / a])                                 # (w1, w2) = (a, 2/a): 모두 같은 함수
    print(f"w1 = {a:5.2f}, w2 = {2 / a:5.2f}  손실 {L2(w):.0e}  Hessian 최대 고윳값 {np.linalg.eigvalsh(num_hess(L2, w))[-1]:6.2f}")

def L1(w):                                                   # w = -1: 깊고 날카로운 골, w = 1.5: 얕고 평평한 골
    return 0.05 * w**2 + 1 - np.exp(-((w + 1) / 0.15) ** 2) - 0.8 * np.exp(-((w - 1.5) / 0.8) ** 2)
grid = np.linspace(-3, 4, 1401)                              # (G,)
for rho in [0.0, 0.1, 0.2]:
    sam = np.array([L1(np.linspace(w - rho, w + rho, 41)).max() for w in grid])   # max_{|e|<=rho} L(w+e)
    print(f"rho = {rho}: 반경 안 최대 손실  w=-1에서 {sam[400]:.3f}, w=1.5에서 {sam[900]:.3f}  ->  최소점 w = {grid[np.argmin(sam)]:.2f}")
```

```output
w1 =  0.10, w2 = 20.00  손실 3e-33  Hessian 최대 고윳값  70.00
w1 =  1.00, w2 =  2.00  손실 0e+00  Hessian 최대 고윳값   0.88
w1 =  1.41, w2 =  1.41  손실 6e-34  Hessian 최대 고윳값   0.70
w1 = 10.00, w2 =  0.20  손실 1e-33  Hessian 최대 고윳값  17.51
rho = 0.0: 반경 안 최대 손실  w=-1에서 0.050, w=1.5에서 0.312  ->  최소점 w = -1.00
rho = 0.1: 반경 안 최대 손실  w=-1에서 0.419, w=1.5에서 0.340  ->  최소점 w = 1.44
rho = 0.2: 반경 안 최대 손실  w=-1에서 0.903, w=1.5에서 0.393  ->  최소점 w = 1.44
```

첫 부분에서 $(w_1, w_2) = (a, 2/a)$는 모두 같은 함수 $2\,\mathrm{ReLU}(x)$이고 손실은 0이다. 그런데 Hessian 최대 고윳값은 $a = \sqrt{2}$에서 0.70, $a = 0.1$에서 70으로 100배 차이 난다. "이 해는 날카로우니 일반화가 나쁠 것이다"라는 추론은 매개화를 고정하지 않으면 성립하지 않는다.

::quiz{id="sharpness-reparametrization"}

**SAM의 아이디어.** 그럼에도 평평한 해를 찾으려는 방법은 실제로 쓰인다. Foret 외(2021)의 **SAM**(sharpness-aware minimization)은 반경 $\rho$ 안의 최악 손실을 최소화한다.

$$
\min_\theta\ \max_{\lVert\boldsymbol{\epsilon}\rVert \le \rho} L(\theta + \boldsymbol{\epsilon}).
$$

안쪽 최대화를 1차 근사로 풀면 $\boldsymbol{\epsilon} = \rho\,\nabla L/\lVert\nabla L\rVert$이고, 갱신은 그 섭동한 점의 gradient로 한다. $\theta \leftarrow \theta - \eta\,\nabla L(\theta + \boldsymbol{\epsilon})$. step마다 gradient를 두 번 계산하므로 비용은 약 두 배다. 셀 두 번째 부분은 깊고 날카로운 골($w = -1$)과 얕고 평평한 골($w = 1.5$)이 있는 1차원 함수에서 SAM 목적함수의 최솟점을 격자로 찾는다. $\rho = 0$이면 깊은 골이 최소지만, $\rho = 0.1$만 되어도 날카로운 골의 반경 안 최악 손실(0.419)이 평평한 골(0.340)보다 커져 최소점이 평평한 골로 옮겨 간다. 이 예는 SAM의 **목적함수**가 무엇을 선호하는지 보여 준다. 실제 SAM 갱신은 이 목적함수를 1차 근사로 따라가므로, 날카로운 골에서 반드시 빠져나온다는 보장은 없다.

SAM은 여러 이미지 분류 과제에서 일반화를 개선했다고 보고되었다(Foret 외, 2021). 다만 그 이유가 정말 "평탄함"인지는 앞의 매개화 문제 때문에 여전히 연구 대상이다. 연구에서 평탄함을 주장할 때는 어떤 매개화에서, 어떤 척도(최대 고윳값, trace, 반경 안 최악 손실)로 쟀는지를 함께 밝힌다.

## 학습 동역학

**큰 학습률의 효과.** 10.3장의 안정 조건 $\eta < 2/L$을 뒤집어 읽으면, 학습률 $\eta$의 경사하강은 sharpness가 $2/\eta$보다 큰 최솟점에 **머물 수 없다.** 그런 점 근처에 가면 가장 가파른 방향으로 진동이 커져 튕겨 나간다. 그래서 학습률은 속도만이 아니라 **어떤 최솟점에 도착하는지**를 정한다. 최솟점이 곡선 $uv = 1$ 전체인 가장 단순한 비볼록 함수 $f(u, v) = (uv - 1)^2/2$에서 확인한다. 최솟점 $(u, v)$의 sharpness는 $u^2 + v^2$이므로, 같은 손실 0의 해들 중에서도 $u = v = 1$이 가장 평평하고(sharpness 2) 한쪽으로 치우칠수록 날카롭다.

```python {exec="browser" session="landscape"}
def grad_uv(p):                                              # f(u, v) = (u v - 1)^2 / 2
    return (p[0] * p[1] - 1) * np.array([p[1], p[0]])
def sharpness(p):                                            # Hessian 최대 고윳값
    u, v = p
    return np.linalg.eigvalsh(np.array([[v * v, 2 * u * v - 1], [2 * u * v - 1, u * u]]))[-1]

for lr in [0.01, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25]:
    p = np.array([0.2, 4.0])                                 # 가장 가까운 최솟점의 sharpness는 약 16
    with np.errstate(all="ignore"):
        for _ in range(5000):
            p = p - lr * grad_uv(p)
    if np.isfinite(p).all():
        print(f"lr = {lr:4}  2/lr = {2 / lr:6.1f}  최종 f = {(p[0] * p[1] - 1) ** 2 / 2:.0e}"
              f"  도착점 ({p[0]:+.2f}, {p[1]:+.2f})  sharpness {sharpness(p):5.2f}")
    else:
        print(f"lr = {lr:4}  2/lr = {2 / lr:6.1f}  발산")
```

```output
lr = 0.01  2/lr =  200.0  최종 f = 6e-32  도착점 (+0.25, +4.00)  sharpness 16.08
lr =  0.1  2/lr =   20.0  최종 f = 2e-32  도착점 (+0.25, +4.00)  sharpness 16.07
lr = 0.12  2/lr =   16.7  최종 f = 0e+00  도착점 (+0.25, +3.99)  sharpness 16.02
lr = 0.15  2/lr =   13.3  최종 f = 0e+00  도착점 (+0.32, +3.09)  sharpness  9.67
lr = 0.18  2/lr =   11.1  최종 f = 0e+00  도착점 (+0.63, +1.59)  sharpness  2.93
lr =  0.2  2/lr =   10.0  최종 f = 2e-32  도착점 (-1.10, -0.91)  sharpness  2.04
lr = 0.25  2/lr =    8.0  발산
```

시작점 $(0.2, 4)$에서 가장 가까운 최솟점은 sharpness가 약 16이다. 학습률이 0.12 이하이면($2/\eta \ge 16.7$) 경사하강은 그 근처의 해 $(0.25, 4.00)$에 도착한다. 학습률이 0.15가 되면 $2/\eta = 13.3$이라 그 해에 머물 수 없고, 튕겨 나간 끝에 sharpness 9.67인 해에 도착한다. 0.18에서는 2.93, 0.2에서는 거의 가장 평평한 2.04다. 모든 경우에 도착한 해의 sharpness는 $2/\eta$ 아래다. 0.25에서는 시작점 자체가 너무 가팔라 발산했다. 같은 손실 0의 해라도 **학습률이 클수록 평평한 해가 선택된다.**

**edge of stability의 직관.** 위 예에서 경사하강은 sharpness가 $2/\eta$ 아래인 곳으로 "밀려났다". Cohen 외(2021)는 신경망을 full-batch 경사하강으로 학습할 때 이보다 더 미묘한 현상이 흔하다고 보고했다. 학습 초기에는 sharpness가 점점 커지다가(progressive sharpening) $2/\eta$ 근처에 도달하면 더 커지지 않고 그 근처를 맴돈다. 그 구간에서 손실은 짧은 시간 척도에서는 오르내리지만, 긴 시간 척도에서는 계속 내려간다. 이를 **edge of stability**라 부른다. 10.3장의 이차 근사로는 이 구간에서 발산해야 하므로, 왜 학습이 무너지지 않고 계속되는지는 이차 근사 밖의 효과로 설명하려는 연구가 이어지고 있다. 실습에서 이 현상을 작은 MLP로 직접 본다.

연구에서의 의미는 두 가지다. 첫째, 학습률을 바꾸면 최적화 속도와 함께 **도착하는 해의 성질**도 바뀐다. 학습률이 다른 두 실험의 일반화 차이를 optimizer의 "성능" 차이로만 해석하지 않는다. 둘째, 10.4장의 warm-up은 학습 초기 sharpness가 큰 영역에서 학습률이 안정 조건을 크게 깨지 않게 하는 장치로도 해석할 수 있다.

## 손실 지형 분석

손실 지형은 수백만 차원이라 직접 볼 수 없다. 대신 낮은 차원의 단면이나 요약 통계를 잰다. 세 도구를 차례로 본다.

**1차원·2차원 단면.** 가장 단순한 것은 두 해 $\theta_a$, $\theta_b$를 잇는 선분 위의 손실 $L\big((1 - t)\theta_a + t\theta_b\big)$이다. Goodfellow 외(2015)는 초기값에서 학습된 해까지의 선분 위에서 손실이 대체로 단조롭게 줄어든다는 관찰로, 학습 경로에 큰 장벽이 없다는 해석을 내놓았다. 해 주변에서 무작위 방향 $\mathbf{d}_1, \mathbf{d}_2$로 자른 2차원 단면 $L(\theta + a\mathbf{d}_1 + b\mathbf{d}_2)$을 등고선으로 그리기도 한다. 이때 방향의 길이를 어떻게 정하느냐가 그림의 모양을 크게 바꾼다. Li 외(2018)는 방향 벡터를 필터마다 해의 필터 노름에 맞춰 정규화하는 filter normalization으로, 서로 다른 신경망의 단면을 공정하게 비교하는 방법을 제안했다.

**mode connectivity.** 서로 다른 시드로 학습한 두 해를 잇는 **직선** 위에서는 손실이 크게 오르는 경우가 많다. 그런데 Garipov 외(2018)는 두 해를 손실이 거의 오르지 않는 **곡선**(예: 꺾인 선, Bézier 곡선)으로 이을 수 있음을 보였다. 서로 떨어진 골짜기처럼 보이던 해들이 실제로는 낮은 손실의 길로 연결되어 있다는 뜻이다. 직선 경로의 장벽은 대칭 때문에 생기기도 한다. 은닉 뉴런의 순서만 바꾼 해는 같은 함수인데도, 둘을 잇는 직선의 중간점은 서로 다른 뉴런의 가중치를 평균한 다른 함수가 된다.

**Hessian 스펙트럼.** 해의 곡률은 Hessian의 고윳값 분포로 요약한다. 신경망에서 Hessian을 만들 수는 없지만, Hessian-vector product(8.4장)는 gradient 계산의 몇 배 비용으로 얻을 수 있다. 이것만으로 power iteration을 돌리면 절댓값이 가장 큰 고윳값을, Lanczos 방법을 쓰면 스펙트럼의 대략적인 모양을 추정할 수 있다. 여기서는 Hessian-vector product를 gradient의 중앙 차분 $\big(\nabla L(\theta + h\mathbf{v}) - \nabla L(\theta - h\mathbf{v})\big)/2h$로 근사한다.

셀 2에서 학습한 MLP 해로 두 도구를 써 본다.

```python {exec="browser" session="landscape"}
def path_losses(a, b, ts=np.linspace(0, 1, 11)):
    return np.array([loss_grad((1 - t) * a + t * b)[0] for t in ts])   # (T,)

perm = np.random.default_rng(1).permutation(H)
idx = np.concatenate([perm, H + perm, 2 * H + perm, [3 * H]])
th_a, th_b = sols[0], sols[1]
th_p = th_a[idx]                                            # 은닉 뉴런 순서만 바꾼 같은 함수
for name, b in [("시드 0 -> 시드 1     ", th_b), ("시드 0 -> 뉴런 순서 바꾼 시드 0", th_p)]:
    Lp = path_losses(th_a, b)
    print(f"{name}  양 끝 {Lp[0]:.4f}, {Lp[-1]:.4f}  경로 최대 {Lp.max():.4f}  장벽 {Lp.max() - max(Lp[0], Lp[-1]):.4f}")

def top_eig(th, iters=100, h=1e-5):
    v = np.random.default_rng(0).normal(size=th.size)
    for _ in range(iters):                                  # power iteration, Hv는 gradient 차분으로
        v = v / np.linalg.norm(v)
        Hv = (loss_grad(th + h * v)[1] - loss_grad(th - h * v)[1]) / (2 * h)
        lam = v @ Hv
        v = Hv
    return lam
for s in [0, 1]:
    print(f"시드 {s} 해의 Hessian 최대 고윳값 {top_eig(sols[s]):.3f}   (학습률 0.2의 안정 한계 2/lr = 10)")
```

```output
시드 0 -> 시드 1       양 끝 0.0053, 0.0078  경로 최대 3.4050  장벽 3.3972
시드 0 -> 뉴런 순서 바꾼 시드 0  양 끝 0.0053, 0.0053  경로 최대 1.7390  장벽 1.7337
시드 0 해의 Hessian 최대 고윳값 6.603   (학습률 0.2의 안정 한계 2/lr = 10)
시드 1 해의 Hessian 최대 고윳값 8.131   (학습률 0.2의 안정 한계 2/lr = 10)
```

서로 다른 시드의 두 해는 양 끝 손실이 0.005–0.008인데 직선 경로의 최대 손실은 3.4다. 같은 해의 뉴런 순서만 바꾼 해, 즉 **같은 함수**와의 직선 경로에도 1.73의 장벽이 있다. 선형 경로의 장벽이 크다는 것만으로 "두 해가 본질적으로 다르다"고 결론 내릴 수 없다는 뜻이다. 두 해의 Hessian 최대 고윳값은 6.6과 8.1로, 학습률 0.2의 안정 한계 $2/\eta = 10$ 아래다. 앞 절의 결론대로, 학습률이 머물 수 있는 해만 남았다.

::quiz{id="permutation-barrier"}

:::note[분야 예시]
기계학습 원자간 퍼텐셜(machine-learned interatomic potential)이나 기후 에뮬레이터처럼 같은 구조의 모델을 시드만 바꿔 여러 개 학습하는 경우가 많다. 그 앙상블의 예측 차이를 불확실성으로 쓰기도 한다(21.1장). 이때 시드마다 다른 골짜기에 도착한다는 것이 앙상블 다양성의 원천이다. 반대로 같은 초기값에서 데이터 순서만 바꾼 모델들은 실습에서 보듯 거의 같은 점으로 갈 수 있어, 앙상블로서의 다양성이 작다. 앙상블을 만들 때 무엇을 바꿨는지가 불확실성 추정의 의미를 정한다.
:::

## 실습

학습된 두 모델 사이의 선형 보간 경로에서 손실을 재고, Hessian 최대 고윳값을 추정한다. 코드는 `labs/optimization/nonconvex-landscape/interpolation_hessian.py`다. 2차원 입력의 회귀 문제 $y = \sin(2x_1)\cos(x_2) + \varepsilon$(256개 점)을 은닉 뉴런 32개의 tanh MLP로 맞춘다.

1. momentum SGD로 세 해를 학습한다. A와 B는 초기값이 다르고, C는 A와 같은 초기값에서 mini-batch 순서만 다르다. A와 B, A와 C, A와 뉴런 순서만 바꾼 A' 사이의 선형 경로에서 장벽을 잰다.
2. 같은 초기값에서 학습률만 바꾼 full-batch 경사하강을 10000 step 돌리며, Hessian 최대 고윳값을 power iteration으로 재고 $2/\eta$와 비교한다.
3. 해 A 주변의 무작위 2차원 단면을 `outputs/slice2d.png`로 저장한다.

집필 시점의 실행 결과는 다음과 같다.

```output
1) 선형 보간 경로 (MLP 은닉 32, momentum SGD lr 0.05, batch 32, 300 epoch)
   경로                                  L(0)     L(1)     경로 최대   장벽(최대 - 양 끝 큰 값)
   다른 시드 (A -> B)                     0.0035  0.0036     0.4781     0.4745
   같은 초기값, 다른 batch 순서 (A -> C)       0.0035  0.0036     0.0036     0.0000
   뉴런 순서만 바꾼 같은 함수 (A -> A')          0.0035  0.0035     0.6053     0.6018
   |A - B| = 8.53,  |A - C| = 0.07,  |A - A'| = 8.47
   A 경로 손실(t = 0, 0.1, ..., 1, 다른 시드): [0.003 0.019 0.085 0.206 0.357 0.467 0.449 0.298 0.128 0.027 0.004]

2) full-batch 경사하강의 sharpness(Hessian 최대 고윳값)와 2/lr, 같은 초기값
   lr     2/lr          0     100     300    1000    3000   10000   최종 손실
   0.02   100.0       7.02    6.92    6.93    7.21    8.10    9.07   0.0051
   0.05    40.0       7.02    6.92    7.09    7.93    8.86    9.84   0.0035
   0.1     20.0       7.02    6.98    7.47    8.56    9.38   10.65   0.0028
   0.2     10.0       7.02    7.20    8.10    9.06   10.03   10.00   0.0052
   0.4      5.0   발산

3) 무작위 2차원 단면: 중심 손실 0.0035, 단면 안 최소 0.0029, 최대 0.0455 -> outputs/slice2d.png
```

결과를 읽는 법은 네 가지다.

- **다른 시드의 해 사이에는 큰 장벽이 있고, 같은 초기값의 해 사이에는 없다.** A와 B 사이의 경로 최대 손실은 양 끝의 100배가 넘는다. 같은 초기값에서 batch 순서만 바꾼 C는 A와의 거리가 0.07로, 다른 시드 사이의 거리(8.5)보다 100배 이상 가깝고 경로 위 장벽도 없다. 이 작은 모델과 학습 설정에서는 초기값이 도착하는 골짜기를 거의 정했다. 더 큰 모델이나 다른 학습 설정에서도 그런지는 따로 확인해야 한다.
- **같은 함수 사이에도 장벽이 있다.** A와 A'은 같은 함수인데 직선 경로의 장벽(0.60)이 다른 시드 사이의 장벽(0.47)보다 크다. A–B 사이 장벽 중 얼마가 이런 대칭 때문인지는 이 실험으로 알 수 없다.
- **sharpness는 학습 중 커지다가 $2/\eta$에서 멈춘다.** 학습률 0.2($2/\eta = 10$)에서 sharpness는 7.0에서 시작해 3000 step에 10.03, 10000 step에 10.00으로 $2/\eta$에 붙었다. 학습률이 작은 0.02–0.1에서는 sharpness가 10000 step 동안 계속 커졌지만 아직 $2/\eta$(20–100)에 한참 못 미친다. 학습률 0.4에서는 초기 sharpness 7.0이 이미 $2/\eta = 5$를 넘어 발산했다. progressive sharpening과 edge of stability의 작은 예다.
- **2차원 단면의 중심이 최솟점이 아니었다.** 단면 안에서 중심(0.0035)보다 낮은 손실(0.0029)이 있다. mini-batch SGD가 멈춘 점은 전체 손실의 정확한 최솟점이 아니고, 무작위 두 방향만으로도 더 낮은 점이 보인다. 반대로 단면이 그릇 모양이라고 해서 나머지 $129 - 2$개 방향이 모두 위로 휜다는 보장도 없다.

이어서 두 연습문제를 푼다.

::exercise{id="top-hessian-eigenvalue"}

::exercise{id="interpolation-barrier"}

## 실패 사례

- **2차원 단면 시각화를 전체 지형으로 해석하는 것.** 무작위 두 방향은 수백만 차원 중 두 개이고, 소수의 큰 곡률 방향이나 음의 곡률 방향과 거의 직교한다. 단면이 매끈한 그릇 모양이어도 해가 최솟점이라는 증거가 아니고, 단면에 다른 골이 없다고 전역 최소라는 증거도 아니다. 방향의 정규화 방식에 따라 같은 해가 날카롭게도 평평하게도 보인다.
- **gradient가 작은 평평한 구간을 수렴으로 판단하는 것.** 안장점 근처의 정체일 수 있다. 학습을 더 돌리거나 Hessian 최소 고윳값의 부호를 확인한다.
- **sharpness 값만으로 일반화를 판단하는 것.** 같은 함수도 매개화에 따라 sharpness가 100배 달라질 수 있다. 서로 다른 구조나 매개화의 모델을 sharpness로 비교하지 않는다.
- **학습률이 다른 두 실험의 차이를 optimizer의 우열로만 보는 것.** 학습률은 도착하는 해의 sharpness에도 상한을 건다. 결과 차이가 최적화 속도 때문인지, 다른 종류의 해에 도착했기 때문인지 구분한다.

## 통과 기준

- 학습이 정체되거나 실패한 사례 하나를 골라, 안장점 정체·초기화·학습률이 만드는 sharpness 상한 중 어느 것으로 설명되는지 손실 곡선과 Hessian 최대 고윳값(power iteration)으로 근거를 들어 쓴다.
- 서로 다른 시드로 학습한 두 모델 사이의 선형 보간 장벽을 재고, 장벽이 대칭 때문일 수 있다는 점을 포함해 해석한다.
- 연습문제 두 개(`top-hessian-eigenvalue`, `interpolation-barrier`)의 테스트를 통과한다.

## 참고 문헌

- Dauphin, Pascanu, Gulcehre, Cho, Ganguli, Bengio (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS 27*, 2933–2941. arXiv:1406.2572. — 고차원 비볼록 최적화에서 안장점의 역할
- Goodfellow, Vinyals, Saxe (2015). Qualitatively characterizing neural network optimization problems. *ICLR 2015*. arXiv:1412.6544. — 선형 보간 경로로 손실 지형을 살피는 방법
- Keskar, Mudigere, Nocedal, Smelyanskiy, Tang (2017). On large-batch training for deep learning: Generalization gap and sharp minima. *ICLR 2017*. arXiv:1609.04836. — 큰 batch, 날카로운 최솟점, 일반화
- Dinh, Pascanu, Bengio, Bengio (2017). Sharp minima can generalize for deep nets. *ICML 2017*, PMLR 70, 1019–1028. — 매개화에 따라 sharpness가 바뀐다는 반론
- Li, Xu, Taylor, Studer, Goldstein (2018). Visualizing the loss landscape of neural nets. *NeurIPS 31*. arXiv:1712.09913. — filter normalization과 손실 지형 시각화
- Garipov, Izmailov, Podoprikhin, Vetrov, Wilson (2018). Loss surfaces, mode connectivity, and fast ensembling of DNNs. *NeurIPS 31*. arXiv:1802.10026. — mode connectivity
- Foret, Kleiner, Mobahi, Neyshabur (2021). Sharpness-aware minimization for efficiently improving generalization. *ICLR 2021*. arXiv:2010.01412. — SAM
- Cohen, Kaur, Li, Kolter, Talwalkar (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR 2021*. arXiv:2103.00065. — progressive sharpening과 edge of stability
