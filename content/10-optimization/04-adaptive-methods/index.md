---
id: optimization/adaptive-methods
title: "적응적 방법과 학습 레시피"
title_en: "Adaptive Methods and Training Recipes"
tier: core
status: draft
prerequisites: [optimization/first-order-methods]
---

# 적응적 방법과 학습 레시피

10.3장의 결론은 "학습률은 곡률이 정한다"였다. 그런데 신경망의 파라미터는 수백만 개이고, 층마다, 파라미터마다 gradient의 크기가 몇 자릿수씩 다르다. 하나의 학습률은 가장 가파른 방향에 맞춰야 하므로, 나머지 방향은 거의 움직이지 못한다. Adam 같은 **적응적 방법**(adaptive method)은 파라미터마다 지금까지의 gradient 크기를 기억해 step 크기를 따로 정한다. 현재 딥러닝에서 가장 널리 쓰이는 기본 optimizer가 이 계열이다.

이 장의 나머지 절반은 optimizer 주변의 **학습 레시피**다. weight decay를 어떻게 넣는지, 학습률을 시간에 따라 어떻게 바꾸는지, gradient clipping과 누적을 언제 쓰는지, batch 크기를 바꾸면 학습률을 어떻게 바꾸는지를 다룬다. 마지막으로 새 optimizer가 나왔을 때 그 주장을 평가하는 관점과, optimizer를 공정하게 비교하는 실험 설계를 다룬다. 레시피의 세부는 빠르게 바뀌지만, 각 기법이 해결하려는 문제는 10.3장의 곡률·잡음 분석으로 설명된다.

## 학습 목표

- Adam 계열의 원리를 설명한다
- weight decay와 L2 정규화를 구분한다
- 학습률 스케줄과 실전 기법을 사용한다

## 적응적 방법

**왜 필요한가.** 10.3장에서 경사하강의 반복 수는 조건수 $\kappa$에 비례했다. 곡률을 알면 각 방향의 step을 곡률로 나눠 조건수를 없앨 수 있지만(10.5장의 Newton 방법), 큰 모델에서는 Hessian을 만들 수 없다. 적응적 방법은 더 싼 대용품을 쓴다. 파라미터마다 **과거 gradient의 크기**를 추적하고, 그것으로 step을 나눈다. 이는 좌표별 스케일만 조정하는 **대각 전처리**(diagonal preconditioning)다.

**AdaGrad.** Duchi 외(2011)의 AdaGrad는 제곱 gradient를 누적한다.

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \qquad \mathbf{x}_t = \mathbf{x}_{t-1} - \eta\,\frac{\mathbf{g}_t}{\sqrt{\mathbf{s}_t} + \epsilon}.
$$

제곱·나눗셈·제곱근은 성분별이다. 자주, 크게 갱신된 좌표는 step이 작아지고, 드물게 갱신되는 좌표(예: 희소한 입력 특성)는 상대적으로 큰 step을 받는다. 단점은 누적합이 계속 커져 유효 학습률이 단조 감소한다는 것이다. 볼록 문제에서는 장점이지만, 긴 비볼록 학습에서는 학습이 너무 일찍 멈출 수 있다.

**RMSProp.** 누적합 대신 지수이동평균(exponential moving average, EMA)을 쓴다. 강의 자료를 통해 널리 알려진 방법이다.

$$
\mathbf{v}_t = \rho\,\mathbf{v}_{t-1} + (1 - \rho)\,\mathbf{g}_t^2, \qquad \mathbf{x}_t = \mathbf{x}_{t-1} - \eta\,\frac{\mathbf{g}_t}{\sqrt{\mathbf{v}_t} + \epsilon}.
$$

$\mathbf{v}$는 최근 약 $1/(1 - \rho)$ step 동안의 제곱 gradient 평균이다. 오래된 정보를 잊으므로 유효 학습률이 0으로 줄지 않는다.

**Adam.** Kingma와 Ba(2015)의 Adam은 RMSProp의 분모에 momentum의 분자를 결합하고, **bias correction**을 더한다.

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t, &
\mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2, \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, &
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t}, \qquad
\mathbf{x}_t = \mathbf{x}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
\end{aligned}
$$

논문의 기본값은 $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$이다. $\mathbf{m}$과 $\mathbf{v}$는 0에서 시작하므로 초기 몇 step 동안 0 쪽으로 치우친다. $\mathbf{m}_t$는 $(1 - \beta_1^t)$만큼, $\mathbf{v}_t$는 $(1 - \beta_2^t)$만큼 작게 추정되고, 이를 나눠 주는 것이 bias correction이다. 보정 계수는 $t$가 커지면 1로 가므로 학습 초기에만 의미가 있다.

**직관.** gradient가 일정하면 $\hat{\mathbf{m}} \approx \mathbf{g}$, $\sqrt{\hat{\mathbf{v}}} \approx \lvert \mathbf{g} \rvert$이므로 갱신은 $\eta\,\operatorname{sign}(\mathbf{g})$에 가깝다. 즉 Adam의 학습률은 "좌표마다 한 step에 움직일 수 있는 대략적인 거리"다. gradient의 부호가 자주 바뀌거나 잡음이 크면 $\lvert \hat{\mathbf{m}} \rvert$가 $\sqrt{\hat{\mathbf{v}}}$보다 작아져 step이 줄어든다. 그리고 gradient 전체에 상수를 곱해도 갱신이 바뀌지 않는다(스케일 불변성).

구현은 수식과 1:1로 대응한다. 두 좌표의 gradient 크기가 $10^4$배 다를 때 첫 step의 이동량을 비교한다.

```python {exec="browser" session="adaptive"}
import numpy as np

def make(name, lr, beta=0.9, rho=0.999, eps=1e-8, bias_correction=True):
    s = {"m": 0.0, "v": 0.0, "t": 0}
    def step(x, g):
        s["t"] += 1
        if name == "momentum":
            s["m"] = beta * s["m"] + g
            return x - lr * s["m"]
        if name == "adagrad":
            s["v"] = s["v"] + g * g                          # 제곱 gradient의 누적합
            return x - lr * g / (np.sqrt(s["v"]) + eps)
        s["v"] = rho * s["v"] + (1 - rho) * g * g            # 제곱 gradient의 EMA
        if name == "rmsprop":
            return x - lr * g / (np.sqrt(s["v"]) + eps)
        s["m"] = beta * s["m"] + (1 - beta) * g              # adam: gradient의 EMA
        m, v = s["m"], s["v"]
        if bias_correction:
            m, v = m / (1 - beta ** s["t"]), v / (1 - rho ** s["t"])
        return x - lr * m / (np.sqrt(v) + eps)
    return step

g = np.array([0.01, 100.0])                                  # 두 좌표의 gradient가 10^4배 다르다
for label, opt in [("momentum", make("momentum", 0.1)), ("AdaGrad", make("adagrad", 0.1)),
                   ("RMSProp(0.99)", make("rmsprop", 0.1, rho=0.99)), ("Adam", make("adam", 0.1)),
                   ("Adam 보정 없음", make("adam", 0.1, bias_correction=False))]:
    print(f"{label:14s} 첫 step 이동량 = {np.abs(opt(np.zeros(2), g))}")
```

```output
momentum       첫 step 이동량 = [1.e-03 1.e+01]
AdaGrad        첫 step 이동량 = [0.0999999 0.1      ]
RMSProp(0.99)  첫 step 이동량 = [0.99999 1.     ]
Adam           첫 step 이동량 = [0.0999999 0.1      ]
Adam 보정 없음     첫 step 이동량 = [0.31621777 0.31622777]
```

momentum SGD는 gradient에 비례해 움직여 두 좌표의 이동량이 $10^4$배 다르다. AdaGrad와 Adam은 두 좌표 모두 정확히 학습률(0.1)만큼 움직인다. bias correction을 빼면 Adam의 첫 step은 $0.1 \times 0.1/\sqrt{0.001} \approx 0.316$으로 학습률의 3.16배다. 보정이 없는 RMSProp($\rho = 0.99$)의 첫 step은 $0.1/\sqrt{0.01} = 1.0$으로 학습률의 10배다. 초기 step이 이렇게 커지는 것을 막는 것이 bias correction의 역할이다.

::quiz{id="adam-first-step"}

::quiz{id="bias-correction-why"}

**대각 전처리의 한계.** 적응적 방법이 조건수를 해결하는 것은 곡률 차이가 **좌표축 방향으로** 갈려 있을 때다. 곡률이 1과 100인 이차함수를, 골짜기가 좌표축과 나란한 경우와 45도 회전한 경우로 나누어 비교한다. optimizer마다 학습률을 격자에서 따로 골랐다.

```python {exec="browser" session="adaptive"}
def landscape(theta):
    c, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s_], [s_, c]])
    return R @ np.diag([1.0, 100.0]) @ R.T                   # (2, 2) Hessian, kappa = 100

def steps_to(name, lr, H, tol=1e-6, max_steps=3000):
    opt, x = make(name, lr), np.array([0.8, -0.5])
    for k in range(1, max_steps + 1):
        x = opt(x, H @ x)
        if not np.isfinite(x).all():
            return np.inf
        if 0.5 * x @ H @ x < tol:
            return k
    return np.inf

lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
with np.errstate(over="ignore", invalid="ignore"):
    for tname, theta in [("축 정렬  ", 0.0), ("45도 회전", np.pi / 4)]:
        H = landscape(theta)
        for name in ["momentum", "adagrad", "rmsprop", "adam"]:
            ks = [steps_to(name, lr, H) for lr in lrs]       # optimizer마다 학습률을 따로 고른다
            i = int(np.argmin(ks))
            print(f"{tname} {name:8s} 최선 lr {lrs[i]:<6} f < 1e-6까지 {ks[i]:5.0f} step")
```

```output
축 정렬   momentum 최선 lr 0.003  f < 1e-6까지   130 step
축 정렬   adagrad  최선 lr 10.0   f < 1e-6까지     5 step
축 정렬   rmsprop  최선 lr 3.0    f < 1e-6까지     4 step
축 정렬   adam     최선 lr 1.0    f < 1e-6까지   122 step
45도 회전 momentum 최선 lr 0.01   f < 1e-6까지   109 step
45도 회전 adagrad  최선 lr 1.0    f < 1e-6까지   375 step
45도 회전 rmsprop  최선 lr 0.03   f < 1e-6까지   348 step
45도 회전 adam     최선 lr 1.0    f < 1e-6까지   133 step
```

축과 나란한 골짜기에서 RMSProp과 AdaGrad는 4–5 step 만에 도달한다. 좌표마다 스케일을 따로 주면 이 문제는 서로 독립된 1차원 문제 두 개가 되기 때문이다. momentum SGD는 130 step이 걸린다. 같은 골짜기를 45도 회전하면 큰 곡률과 작은 곡률이 두 좌표에 섞여 좌표별 스케일로는 분리할 수 없다. RMSProp과 AdaGrad는 350 step 안팎으로 느려지고, 회전과 무관한 momentum SGD(109 step)보다도 느려진다. Adam은 두 경우 모두 120–130 step으로 momentum SGD와 비슷했다. momentum 항($\beta_1 = 0.9$)이 더해져 RMSProp과 다르게 움직인 것으로 보이지만, 이 실험만으로 원인을 단정하지는 않는다. 축 정렬 지형에서 AdaGrad의 최선 학습률이 격자 끝(10)에 있다는 점도 기록해 둔다. 좌표가 분리되면 큰 step이 오히려 유리하기 때문이고, 격자를 더 넓히면 결과가 바뀔 수 있다.

::quiz{id="diagonal-preconditioning"}

아래 위젯에서 같은 지형 위에 여러 optimizer를 동시에 놓고 궤적을 비교한다. 지형을 회전시키며 Adam의 궤적이 어떻게 바뀌는지, 학습률을 바꾸면 Adam이 최솟점 근처에서 학습률 크기의 진동을 보이는지 확인한다.

::widget{name="optimizer-landscape"}

신경망의 손실 지형은 좌표축과 완전히 나란하지도, 완전히 무작위로 회전되어 있지도 않다. 층마다·파라미터 종류마다 gradient 스케일이 크게 다르다는 사실이 좌표별 스케일링이 실제로 도움이 되는 주된 이유로 흔히 설명된다. 반면 적응적 방법이 늘 이기는 것은 아니다. Adam은 간단한 볼록 문제에서도 최적해로 수렴하지 않는 반례가 알려져 있고, 원인은 제곱 gradient의 지수이동평균이 작아지며 유효 step이 커질 수 있다는 점이다(Reddi 외, 2018).

:::note[분야 예시]
물리 모델의 파라미터를 데이터로 맞출 때 파라미터의 단위와 크기가 크게 다른 경우가 흔하다. 예를 들어 반응–확산 모델에서 확산계수는 $10^{-9}\,\mathrm{m^2/s}$ 수준이고 반응 속도 상수는 $10^{3}\,\mathrm{s^{-1}}$ 수준일 수 있다. Adam은 좌표마다 한 step에 약 $\eta$만큼 움직이므로, $\eta = 10^{-3}$이면 확산계수가 한 step에 자기 크기의 $10^6$배만큼 움직이려 한다. gradient 스케일은 보정되지만 **파라미터 스케일**은 보정되지 않는다. 그래서 적응적 optimizer를 쓰더라도 파라미터를 무차원화하거나 로그 스케일($\log D$)로 다시 매개화해 크기를 맞추는 것이 먼저다.
:::

## weight decay

**L2 정규화와 weight decay.** 두 용어는 흔히 섞여 쓰이지만 정의가 다르다.

- **L2 정규화**: 손실에 $\frac{\lambda}{2}\lVert \mathbf{w} \rVert^2$를 더한다. gradient는 $\nabla L(\mathbf{w}) + \lambda\mathbf{w}$가 된다. 10.2장에서 본 노름 제약과 동치인 정규화다.
- **weight decay**: 매 step 파라미터를 일정 비율로 줄인다. $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta\,(\text{optimizer의 갱신 방향})$.

SGD에서는 둘이 같다. $\mathbf{w} - \eta(\nabla L + \lambda\mathbf{w}) = (1 - \eta\lambda)\mathbf{w} - \eta\nabla L$이기 때문이다. Adam에서는 다르다. L2 정규화의 $\lambda\mathbf{w}$는 gradient에 섞여 $\sqrt{\hat{\mathbf{v}}}$로 함께 나뉘므로, 과거 gradient가 큰 좌표일수록 감쇠가 약해진다. Loshchilov와 Hutter(2019)는 이를 지적하고, 감쇠를 적응적 스케일링 밖으로 빼낸 **decoupled weight decay**와 그것을 Adam에 적용한 **AdamW**를 제안했다.

$$
\mathbf{w}_t = \mathbf{w}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta\lambda\,\mathbf{w}_{t-1}.
$$

곡률(gradient 스케일)이 0.01, 1, 100으로 다른 세 좌표의 이차 손실 $\sum_i a_i(w_i - 1)^2/2$에 $\lambda = 1$로 두 방식을 적용한다.

```python {exec="browser" session="adaptive"}
a = np.array([0.01, 1.0, 100.0])                             # 좌표별 곡률(gradient 스케일)
grad = lambda w: a * (w - 1.0)                               # L(w) = sum a_i (w_i - 1)^2 / 2
lam, lr = 1.0, 0.01

w_l2, w_wd = np.zeros(3), np.zeros(3)
for _ in range(2000):                                        # SGD: 두 방식이 같은가?
    w_l2 = w_l2 - lr * (grad(w_l2) + lam * w_l2)
    w_wd = w_wd - lr * grad(w_wd) - lr * lam * w_wd
print(f"SGD   L2 vs decoupled 최대 차이: {np.abs(w_l2 - w_wd).max():.1e}")

def adam(decoupled, steps=20000, b1=0.9, b2=0.999, eps=1e-8):
    w, m, v = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in range(1, steps + 1):
        g = grad(w) if decoupled else grad(w) + lam * w      # L2: 감쇠 항이 gradient에 들어간다
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        u = (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        w = w - lr * u - (lr * lam * w if decoupled else 0.0)
    return w

print("L2 정규화 문제의 해 a/(a+lam):", (a / (a + lam)).round(4))
print("Adam + L2              :", adam(False).round(4))
print("AdamW (decoupled)      :", adam(True).round(4))
```

```output
SGD   L2 vs decoupled 최대 차이: 8.3e-16
L2 정규화 문제의 해 a/(a+lam): [0.0099 0.5    0.9901]
Adam + L2              : [0.0099 0.5    0.9901]
AdamW (decoupled)      : [0.9514 0.9514 0.9514]
```

SGD에서 두 방식의 차이는 반올림 오차 수준이다. Adam + L2는 L2 정규화 문제의 해 $a_i/(a_i + \lambda)$로 정확히 간다. gradient가 큰 좌표($a = 100$)는 거의 줄지 않고(0.9901), 작은 좌표($a = 0.01$)는 거의 0이 된다. AdamW는 세 좌표를 모두 같은 값(0.9514)으로 보낸다. AdamW의 결과는 어떤 L2 정규화 문제의 해도 아니다. 감쇠의 세기가 손실의 모양이 아니라 학습률과 step 수로 정해진다.

::quiz{id="adam-l2-vs-adamw"}

어느 쪽이 "옳은" 정규화인지는 문제에 달려 있다. 다만 두 가지는 분명하다. 첫째, 둘은 다른 알고리즘이므로 논문과 코드에서 어느 것을 썼는지 밝혀야 한다. 둘째, 라이브러리의 인자 이름을 믿지 말고 문서를 확인한다. 집필 시점 예시로, PyTorch의 `torch.optim.Adam`의 `weight_decay` 인자는 L2 항을 gradient에 더하는 방식이고, decoupled 방식은 `torch.optim.AdamW`다. 현재 버전의 동작은 각 optimizer 문서의 알고리즘 설명에서 확인한다.

::quiz{id="decay-transfer"}

## 학습률 스케줄

10.3장에서 고정 학습률 SGD는 잡음 바닥에 머물고, 학습률을 줄이면 바닥이 내려갔다. 학습률 스케줄(learning rate schedule)은 이 관찰을 체계화한 것이다. 초반에는 큰 학습률로 빠르게 이동하고, 후반에는 작은 학습률로 잡음을 줄여 정밀하게 수렴한다.

- **step decay**: 정해진 step(또는 epoch)마다 학습률에 0.1 같은 상수를 곱한다.
- **cosine**: 총 $T$ step 동안 $\eta_k = \frac{\eta_{\max}}{2}\big(1 + \cos(\pi k/T)\big)$로 부드럽게 0까지 줄인다. Loshchilov와 Hutter(2017)가 주기적 재시작과 함께 제안한 형태에서 재시작을 뺀 것이다.
- **warm-up**: 처음 수백~수천 step 동안 학습률을 0 근처에서 목표값까지 선형으로 올린다. 보통 다른 스케줄 앞에 붙인다.
- **one-cycle**: 학습률을 낮은 값에서 최댓값까지 올렸다가 다시 아주 작은 값까지 내리는 한 번의 주기. 여기서는 선형으로 올리고 내리는 단순화한 형태를 쓴다(구현마다 세부가 다르다).

```python {exec="browser" session="adaptive"}
T, lr_max, warm = 1000, 0.1, 50
def step_decay(k):
    return lr_max * 0.1 ** (k // 400)
def cosine(k):
    return 0.5 * lr_max * (1 + np.cos(np.pi * k / T))
def warmup_cosine(k):
    if k < warm:
        return lr_max * (k + 1) / warm
    return 0.5 * lr_max * (1 + np.cos(np.pi * (k - warm) / (T - warm)))
def one_cycle(k, peak=300):                                  # 선형 증가 후 선형 감소(단순화한 형태)
    return lr_max * (0.1 + 0.9 * k / peak) if k < peak else lr_max * (1 - (k - peak) / (T - peak))

print("step   " + "".join(f"{k:>9d}" for k in [0, 25, 50, 300, 400, 500, 800, 999]))
for name, fn in [("step", step_decay), ("cosine", cosine), ("warm+cos", warmup_cosine), ("1cycle", one_cycle)]:
    print(f"{name:8s}" + "".join(f"{fn(k):9.4f}" for k in [0, 25, 50, 300, 400, 500, 800, 999]))
```

```output
step           0       25       50      300      400      500      800      999
step       0.1000   0.1000   0.1000   0.1000   0.0100   0.0100   0.0010   0.0010
cosine     0.1000   0.0998   0.0994   0.0794   0.0655   0.0500   0.0095   0.0000
warm+cos   0.0020   0.0520   0.1000   0.0839   0.0701   0.0541   0.0105   0.0000
1cycle     0.0100   0.0175   0.0250   0.1000   0.0857   0.0714   0.0286   0.0001
```

**warm-up의 이유.** 학습 초기에는 파라미터가 무작위 초기값 근처에 있고, 그 지점의 곡률이 크거나 Adam의 $\hat{\mathbf{v}}$ 같은 통계가 아직 몇 개의 표본으로만 추정된 상태다. 이때 목표 학습률을 바로 쓰면 첫 몇 step이 손실을 크게 튀게 하거나 발산시킬 수 있다. 큰 batch로 학습률을 비례해 키운 대규모 학습에서, 처음 몇 epoch 동안 학습률을 서서히 올리는 warm-up이 이런 초기 최적화의 어려움을 해결했다는 보고가 대표적이다(Goyal 외, 2017). warm-up이 필요한지, 얼마나 길어야 하는지는 모델과 optimizer에 따라 다르므로 실험으로 확인한다.

::quiz{id="warmup-reason"}

스케줄을 고를 때 기억할 점은 두 가지다. 첫째, 스케줄은 **총 학습 길이 $T$에 묶여 있다.** cosine 스케줄로 1000 step 학습한 결과를 500 step에서 끊으면 학습률이 아직 큰 상태라 같은 결과가 아니다. 학습 길이를 바꾸면 스케줄도 다시 정한다. 둘째, 스케줄과 최대 학습률은 함께 튜닝해야 한다. 스케줄을 바꾸면 최선의 $\eta_{\max}$도 달라진다.

## 실전 기법

**gradient clipping.** 손실 지형의 가파른 절벽이나 드문 이상치 batch 때문에 gradient가 갑자기 커지면 한 step에 파라미터가 멀리 튀어 학습이 망가질 수 있다. **global norm clipping**은 모든 파라미터의 gradient를 이어 붙인 벡터의 노름이 임계값 $c$를 넘으면 전체를 $c/\lVert \mathbf{g} \rVert$배로 줄인다. 방향은 그대로 두고 크기만 제한한다. 순환 신경망처럼 gradient 폭주가 잘 일어나는 모델에서 널리 쓰인다. clipping이 매 step 발동한다면 학습률이 너무 크다는 신호이므로, 발동 빈도를 기록해 둔다.

**gradient accumulation.** 메모리 때문에 원하는 batch 크기를 한 번에 올릴 수 없으면, 작은 micro-batch 여러 개의 gradient를 더한 뒤 한 번 갱신한다. 각 micro-batch의 gradient가 평균 손실에 대한 것이면, 누적한 합을 micro-batch 수로 나눠야 큰 batch의 평균 gradient와 같다. 나누지 않으면 유효 학습률이 그 배수만큼 커진다. micro-batch 크기가 다르면 샘플 수로 가중해야 한다. batch normalization처럼 batch 통계를 쓰는 층은 누적해도 작은 batch의 통계를 쓰므로 큰 batch와 완전히 같아지지는 않는다.

```python {exec="browser" session="adaptive"}
rng = np.random.default_rng(0)
X, y = rng.normal(size=(64, 3)), rng.normal(size=64)        # (N, D), (N,)
w = np.ones(3)
grad_of = lambda Xb, yb: Xb.T @ (Xb @ w - yb) / len(yb)     # 평균 손실의 gradient

g = grad_of(X, y)
for max_norm in [10.0, 0.5]:
    scale = min(1.0, max_norm / np.linalg.norm(g))           # global norm clipping
    gc = g * scale
    cos = gc @ g / (np.linalg.norm(gc) * np.linalg.norm(g))
    print(f"clip {max_norm:4.1f}: |g| {np.linalg.norm(g):.3f} -> {np.linalg.norm(gc):.3f}, 방향 cos = {cos:.3f}")

parts = [slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64)]   # micro-batch 4개
acc_mean = sum(grad_of(X[p], y[p]) for p in parts) / len(parts)
acc_sum = sum(grad_of(X[p], y[p]) for p in parts)
print("누적 후 평균 == 전체 batch gradient :", np.allclose(acc_mean, g))
print("누적 후 합(나누기 생략) / 전체      :", (np.linalg.norm(acc_sum) / np.linalg.norm(g)).round(3))
uneven = [slice(0, 8), slice(8, 64)]                         # 크기가 다른 micro-batch
print("크기가 다른 micro-batch의 단순 평균 오차:", np.linalg.norm(sum(grad_of(X[p], y[p]) for p in uneven) / 2 - g).round(3))
```

```output
clip 10.0: |g| 2.183 -> 2.183, 방향 cos = 1.000
clip  0.5: |g| 2.183 -> 0.500, 방향 cos = 1.000
누적 후 평균 == 전체 batch gradient : True
누적 후 합(나누기 생략) / 전체      : 4.0
크기가 다른 micro-batch의 단순 평균 오차: 0.558
```

clipping은 노름을 2.183에서 0.5로 줄였지만 방향의 cosine은 1로 그대로다. 크기가 같은 micro-batch 4개의 평균은 전체 batch gradient와 일치하고, 나누기를 빠뜨리면 정확히 4배가 된다. 크기가 8과 56인 micro-batch를 단순 평균하면 작은 micro-batch가 과대 대표되어 전체 gradient와 크게 어긋난다.

::quiz{id="accumulation-divisor"}

**batch 크기와 학습률.** batch를 키우면 gradient 잡음의 분산이 $1/B$로 줄어든다(10.3장). Goyal 외(2017)는 momentum SGD로 ImageNet을 학습할 때 batch를 $k$배 키우면 학습률도 $k$배 키우는 **선형 스케일링 규칙**(linear scaling rule)과 warm-up을 함께 써서, batch 8192까지 작은 batch와 비슷한 정확도를 얻었다고 보고했다. 직관적으로는 같은 수의 샘플을 보는 동안 작은 batch로 $k$번 가는 것을 큰 batch로 한 번에 가려면 step이 $k$배 커야 한다. 그러나 이 규칙은 학습률이 곡률 한계($2/L$, 10.3장)에 닿기 전까지만 성립하고, 아주 큰 batch에서는 깨진다. Adam 같은 적응적 방법에는 같은 규칙이 그대로 적용되지 않을 수 있으므로, batch 크기를 바꾸면 학습률을 다시 튜닝하는 것이 안전하다.

## 최신 optimizer의 위치

새 optimizer는 계속 제안된다. 몇 가지 예와, 그것을 읽는 관점을 정리한다(세부 알고리즘과 수치는 각 원문에서 확인한다).

- **Lion**(Chen 외, 2023): 프로그램 탐색으로 찾은 optimizer다. momentum만 상태로 두고 갱신에 부호(sign) 연산을 씌워, 모든 좌표가 같은 크기로 움직인다. 2차 모멘트를 저장하지 않아 AdamW보다 메모리를 적게 쓰고, 갱신의 노름이 커서 AdamW보다 작은 학습률이 필요하다고 보고되었다.
- **Shampoo**(Gupta 외, 2018): 행렬·텐서 모양의 파라미터에 대해, 텐서의 각 차원마다 작은 전처리 행렬을 두어 대각보다 풍부한 전처리를 한다. 앞 절의 실험에서 대각 전처리가 회전된 골짜기를 다루지 못했는데, 이런 비대각 구조를 일부 잡으려는 방향이다.
- **Muon**: 2차원 가중치 행렬의 momentum 갱신을 Newton–Schulz 반복으로 근사 직교화해서 쓰는 방법으로, 2024년 블로그와 공개 코드로 알려졌다. 행렬이 아닌 파라미터(임베딩·출력층 등)에는 보통 AdamW를 함께 쓴다.

새 방법을 평가할 때 다음을 확인한다.

1. **무엇을 전처리하는가.** 대각(Adam), 부호(Lion), 차원별 행렬(Shampoo), 직교화(Muon) 중 어느 구조를 이용하는지가 어떤 문제에서 이득이 있을지를 정한다.
2. **비용.** step당 계산량, 메모리(상태 변수의 수), 분산 학습에서의 통신량. 같은 시간·같은 메모리로 비교했는가.
3. **비교 기준선이 공정한가.** 기준 optimizer(AdamW 등)의 학습률·weight decay·스케줄을 새 방법만큼 튜닝했는가. 다음 절의 문제다.
4. **규모와 과제에 따라 결론이 유지되는가.** 작은 모델에서의 이득이 큰 모델에서도 유지되는지, 한 종류의 과제에서만의 결과인지 본다.

## 공정한 optimizer 비교

optimizer마다 좋은 학습률의 범위가 다르다. 이 장의 두 번째 실험에서도 최선의 학습률이 momentum SGD는 0.003–0.01, Adam은 1.0으로 두 자릿수 이상 차이 났다. 그래서 "모든 optimizer를 학습률 $10^{-3}$으로 비교했다"는 실험은 사실상 "$10^{-3}$이 어느 optimizer에 맞는가"를 잰 것이다. Choi 외(2019)는 신경망 optimizer 비교에서 하이퍼파라미터 탐색 공간이 순위를 정하는 가장 중요한 요인일 수 있고, 탐색 공간을 바꾸면 기존 비교의 순위가 뒤집힐 수 있다고 보고했다.

공정한 비교의 최소 조건은 다음과 같다.

- **optimizer마다 학습률을 따로 튜닝한다.** 로그 간격(예: 반 decade)의 격자를 쓰고, 최선값이 격자의 끝에 있으면 격자를 넓힌다.
- **튜닝 예산을 맞춘다.** 한쪽은 100번, 다른 쪽은 3번 시도했다면 공정하지 않다. 하이퍼파라미터가 많은 optimizer는 그만큼 튜닝 비용이 크다는 점도 비교의 일부다.
- **검증 데이터로 고르고 시험 데이터로 보고한다.** 같은 데이터로 고르고 보고하면 고르는 과정의 운이 결과에 섞인다.
- **시드를 여러 개 쓴다.** 차이가 시드 사이의 변동보다 큰지 확인한다(12.5장).

## 실습

optimizer만 바꾼 비교와 학습률까지 튜닝한 비교를 수행해 결론이 어떻게 달라지는지 본다. 코드는 `labs/optimization/adaptive-methods/fair_comparison.py`다. 1차원 회귀 $y = \sin(2x) + \varepsilon$($\varepsilon$의 분산 0.01)을 은닉 뉴런 32개(tanh)짜리 MLP로 맞추고, NumPy로 직접 역전파한다. momentum SGD($\beta = 0.9$)와 Adam을 3000 step, batch 32로 학습한다.

- **비교 A**: 두 optimizer 모두 학습률 $10^{-3}$(흔히 쓰는 Adam 기본값).
- **비교 B**: optimizer마다 학습률 격자 $10^{-4}$–$1$(반 decade 간격)을 탐색해, 시드 3개의 평균 검증 MSE가 가장 낮은 학습률을 고른다.

두 비교 모두 고른 설정의 시험 MSE를 시드별로 보고한다. 집필 시점의 실행 결과는 다음과 같다.

```output
검증 MSE (시드 3개 평균, 3000 step, batch 32). 잡음 분산 = 0.01
lr            1e-04    3e-04    1e-03    3e-03    1e-02    3e-02    1e-01    3e-01    1e+00
SGD+momentum    0.128    0.077    0.071    0.065    0.058    0.039    0.030      발산      발산
Adam            0.167    0.077    0.065    0.050    0.037    0.030    0.016    0.060    2.892

비교 A: 같은 학습률 1e-3
  SGD+momentum 시험 MSE 0.070 (시드별 [0.069 0.07  0.07 ])
  Adam         시험 MSE 0.064 (시드별 [0.063 0.064 0.064])
비교 B: optimizer마다 학습률을 튜닝
  SGD+momentum 검증으로 고른 lr 1e-01  시험 MSE 0.030 (시드별 [0.032 0.031 0.028])
  Adam         검증으로 고른 lr 1e-01  시험 MSE 0.015 (시드별 [0.012 0.012 0.019])
```

결과를 읽는 법은 세 가지다.

- **비교 A만 보면 "두 optimizer는 비슷하다"는 결론이 나온다.** 시험 MSE 0.070과 0.064는 둘 다 잡음 바닥(0.01)에서 멀고, 차이도 작다. 그러나 이는 두 optimizer 모두에게 너무 작은 학습률에서의 비교다.
- **비교 B에서는 두 optimizer 모두 2–4배 좋아지고, Adam이 SGD의 약 절반 오차로 앞선다.** 결론이 "비슷하다"에서 "이 문제와 이 예산에서는 Adam이 낫다"로 바뀌었다. 학습률 튜닝의 효과(같은 optimizer 안에서 2–4배)가 비교 A에서 본 optimizer 사이의 차이보다 훨씬 크다.
- **이 결과는 이 문제, 3000 step, 이 격자에 대한 것이다.** 두 optimizer의 최선 학습률($10^{-1}$)은 격자 안쪽에 있지만, SGD는 바로 다음 값($3 \times 10^{-1}$)에서 발산하므로 그 사이 값이나 스케줄을 더하면 결과가 달라질 수 있다. step 수를 늘리거나 다른 모델·데이터에서 순위가 유지되는지는 따로 확인해야 한다. 한 실험의 순위를 일반적인 우열로 옮기지 않는다.

이어서 두 연습문제를 푼다.

::exercise{id="adam-step"}

::exercise{id="adamw-decay"}

## 실패 사례

- **Adam에 L2 정규화를 넣고 weight decay라 부르는 것.** 라이브러리에 따라 `weight_decay`라는 인자가 L2 정규화를 구현하기도 한다. Adam에서 두 방식은 다른 해로 가므로(이 장의 세 번째 실험), 논문에는 어느 방식을 썼는지 정확히 쓰고 코드에서 확인한다.
- **다른 optimizer를 같은 학습률로 비교하는 것.** optimizer마다 학습률의 적정 범위가 자릿수 단위로 다르다. 같은 학습률로 비교한 결과는 학습률이 어느 쪽에 맞았는지를 보여 줄 뿐이다. optimizer마다 같은 예산으로 학습률을 튜닝한다.
- **파라미터 스케일을 맞추지 않고 Adam에 맡기는 것.** Adam은 gradient의 스케일은 보정하지만, 학습률이 좌표마다 한 step에 움직이는 거리이므로 파라미터 자체의 크기는 보정하지 못한다. 크기가 몇 자릿수 다른 물리 파라미터는 무차원화나 로그 매개화로 먼저 맞춘다.
- **학습 길이를 바꾸면서 스케줄을 그대로 두는 것.** cosine 같은 스케줄은 총 step 수에 맞춰 정의된다. 짧게 끊은 실험과 끝까지 간 실험을 비교하면 학습률이 다른 상태를 비교하게 된다.

## 통과 기준

- 자신의 학습 설정에서 optimizer·학습률·스케줄·weight decay 방식을 고르고, 각 선택의 근거(곡률·잡음·파라미터 스케일·메모리)를 한 문단으로 쓴다.
- 두 optimizer 이상을 학습률 격자 탐색으로 비교하고, 최선값이 격자 안쪽에 있는지와 시드 간 변동을 함께 보고한다.
- 연습문제 두 개(`adam-step`, `adamw-decay`)의 테스트를 통과한다.

## 참고 문헌

- Duchi, Hazan, Singer (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12, 2121–2159. — AdaGrad
- Kingma, Ba (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980. — Adam과 bias correction
- Reddi, Kale, Kumar (2018). On the convergence of Adam and beyond. *ICLR 2018*. — Adam이 수렴하지 않는 볼록 반례와 AMSGrad
- Loshchilov, Hutter (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*. arXiv:1608.03983. — cosine 학습률 스케줄과 재시작
- Loshchilov, Hutter (2019). Decoupled weight decay regularization. *ICLR 2019*. arXiv:1711.05101. — 적응적 방법에서 L2 정규화와 weight decay의 차이, AdamW
- Goyal 외 (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677. — 선형 스케일링 규칙과 warm-up
- Choi, Shallue, Nado, Lee, Maddison, Dahl (2019). On empirical comparisons of optimizers for deep learning. arXiv:1910.05446. — 하이퍼파라미터 탐색 공간이 optimizer 순위에 미치는 영향
- Gupta, Koren, Singer (2018). Shampoo: Preconditioned stochastic tensor optimization. *ICML 2018*, PMLR 80, 1842–1850. — 텐서 차원별 전처리
- Chen 외 (2023). Symbolic discovery of optimization algorithms. *NeurIPS 2023*. arXiv:2302.06675. — Lion
