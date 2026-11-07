---
id: deep-learning-pytorch/training-loop
title: "학습 루프 해부: forward → loss → backward → update"
title_en: "Anatomy of a Training Loop"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/modules, deep-learning-pytorch/datasets-and-dataloaders, optimization/adaptive-methods, information-and-learning-theory/objective-design]
---

# 학습 루프 해부: forward → loss → backward → update

딥러닝의 학습 루프는 열 줄 남짓이다. 그런데 그 열 줄의 순서를 하나 바꾸거나 한 줄을 빠뜨려도 대부분 오류가 나지 않는다. `zero_grad()`를 빠뜨리면 gradient가 쌓이고, scheduler를 잘못된 주기로 부르면 학습률이 엉뚱하게 줄고, 평가 때 `eval()`을 잊으면 점수가 batch 구성에 따라 흔들린다. 손실 곡선은 그럴듯하게 내려가므로 버그는 결과가 나쁠 때에야, 또는 영영 드러나지 않는다.

이 장은 학습 루프의 모든 줄을 수식과 대응시킨다. 17.2–17.4장에서 텐서·autograd·Module·DataLoader를 따로 보았고, 8.4장에서 gradient 계산을, 10.4장에서 optimizer·스케줄·clipping·accumulation을, 14.3장에서 손실의 설계를 다뤘다. 이 장은 그것들이 한 반복 안에서 **어떤 순서로, 어떤 주기로** 실행되는지에 집중한다. 목표는 학습 루프의 어떤 줄을 가리켜도 그 줄이 하는 일과 대응하는 수식, 그리고 그 줄이 틀렸을 때의 증상을 말할 수 있게 되는 것이다.

## 학습 목표

- end-to-end 학습 과정의 모든 단계를 설명하고 구현한다
- 수식과 코드를 대응시킨다

## 전체 흐름

**왜 필요한가.** 학습 루프의 버그는 대개 단계 사이의 **경계**에서 생긴다. 데이터가 sample에서 batch가 될 때(17.4장), batch가 GPU로 갈 때, gradient가 버퍼에 쌓이고 비워질 때다. 전체 흐름을 한 번 그려 두면 각 버그가 어느 경계의 문제인지 바로 찾을 수 있다.

**정의.** 학습 한 step은 다음 순서로 진행된다.

1. **원본 데이터 → sample**: Dataset의 `__getitem__`이 파일·배열에서 sample 하나를 sample contract에 맞춰 만든다(17.4장).
2. **sample → batch**: DataLoader가 sampler 순서대로 sample을 모아 collate한다.
3. **device 이동**: batch를 모델이 있는 device로 옮긴다.
4. **forward**: $\hat{y} = f_\theta(x)$.
5. **loss**: $L = \ell(\hat{y}, y)$. 스칼라 하나.
6. **gradient 초기화**: `.grad` 버퍼를 비운다(`zero_grad`). 누적할 때는 비우지 않고 다음 backward로 넘어간다.
7. **backward**: $g = \nabla_\theta L$을 계산해 버퍼에 더한다.
8. **update**: $\theta \leftarrow \mathrm{Update}(\theta, g)$. optimizer가 버퍼와 자신의 상태(momentum 등)로 파라미터를 바꾼다. 필요하면 그 전에 clipping, 그 뒤에 scheduler.
9. **주기적으로**: validation(평가 모드, gradient 없음), logging, checkpoint 저장.

아래 위젯은 선형 모델 하나로 4–8단계를 한 단계씩 실행한다. 학습률을 바꿔 발산시켜 보고, zero_grad를 끄면 손실이 어떻게 되는지 확인한다.

::widget{name="training-loop-stepper"}

**코드.** 표준 루프 전체를 쓰면 다음과 같다. 주석이 각 줄에 대응하는 수식이다.

```python {exec="python"}
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.rand(512, 2) * 4 - 2                              # (N, 2) 입력
y = torch.sin(X[:, :1]) * X[:, 1:] + 0.05 * torch.randn(512, 1)   # (N, 1)
train = DataLoader(TensorDataset(X[:384], y[:384]), batch_size=32, shuffle=True)
val_x, val_y = X[384:], y[384:]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(30):
    model.train()
    for xb, yb in train:                                    # batch: (B, 2), (B, 1)
        xb, yb = xb.to(device), yb.to(device)               # device 이동
        pred = model(xb)                                    # forward: y_hat = f_theta(x)
        loss = nn.functional.mse_loss(pred, yb)             # loss: L = l(y_hat, y)
        opt.zero_grad()                                     # .grad를 비운다
        loss.backward()                                     # backward: g = dL/dtheta
        opt.step()                                          # update: theta <- Update(theta, g)
    if epoch % 10 == 9:
        model.eval()
        with torch.no_grad():                               # 평가: 그래프를 만들지 않는다
            val_mae = (model(val_x.to(device)) - val_y.to(device)).abs().mean().item()
        print(f"epoch {epoch + 1:2d}  마지막 batch 손실 {loss.item():.4f}  검증 MAE {val_mae:.4f}")
```

```output
epoch 10  마지막 batch 손실 0.0065  검증 MAE 0.0871
epoch 20  마지막 batch 손실 0.0055  검증 MAE 0.0757
epoch 30  마지막 batch 손실 0.0041  검증 MAE 0.0635
```

출력은 CPU에서 실행한 결과다. 평가 부분에서 `model.eval()`과 `torch.no_grad()`는 하는 일이 다르다. 앞의 것은 dropout·BatchNorm의 **동작**을 바꾸고(17.1장), 뒤의 것은 **그래프를 만들지 않는다**(17.2장). 둘 다 필요하다. 평가가 끝나면 다음 epoch의 첫 줄에서 `model.train()`으로 되돌린다.

## 수식과 코드

**정의.** 학습은 경험적 위험(empirical risk)을 최소화하는 반복이다. batch $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$에 대해

$$
\hat{y}_i = f_\theta(x_i), \qquad L(\theta) = \frac{1}{B}\sum_{i=1}^{B}\ell(\hat{y}_i, y_i), \qquad g = \nabla_\theta L(\theta), \qquad \theta \leftarrow \mathrm{Update}(\theta, g).
$$

$f_\theta$는 모델, $\ell$은 표본 하나의 손실(14.3장에서 보았듯 대개 음의 로그우도), $g$는 batch 평균 손실의 gradient로 전체 데이터 gradient의 확률적 추정량이며(10.3장), Update는 SGD라면 $\theta - \eta g$, AdamW라면 10.4장의 식이다. 코드와의 대응은 다음과 같다.

| 수식 | 코드 | 하는 일 |
|---|---|---|
| $\hat{y} = f_\theta(x)$ | `pred = model(xb)` | forward, 그래프 기록 |
| $L = \frac{1}{B}\sum_i \ell(\hat{y}_i, y_i)$ | `loss = loss_fn(pred, yb)` | batch 평균(`reduction="mean"`이 기본값) |
| (버퍼 초기화) | `opt.zero_grad()` | `.grad`를 비운다 |
| $g = \nabla_\theta L$ | `loss.backward()` | reverse mode로 $g$를 `.grad`에 더한다 |
| $\theta \leftarrow \mathrm{Update}(\theta, g)$ | `opt.step()` | `.grad`와 optimizer 상태로 $\theta$를 바꾼다 |

**코드.** 프레임워크가 숨기는 일을 드러내기 위해, 선형회귀 $\hat{y} = \mathbf{x}^\top\mathbf{w} + b$의 같은 반복을 NumPy로 쓴다. backward 자리에는 8.2장에서 유도한 gradient를 직접 쓴다.

```python {exec="browser" session="training-loop"}
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 3))                                 # (B, D) batch 하나
y = X @ np.array([1.0, -2.0, 0.5]) + 0.3                    # (B,)
w, b, lr = np.zeros(3), 0.0, 0.1                            # theta = (w, b)
for step in range(3):
    y_hat = X @ w + b                                       # y_hat = f_theta(x)         (B,)
    loss = np.mean((y_hat - y) ** 2)                        # L = (1/B) sum (y_hat - y)^2
    r = 2 * (y_hat - y) / len(y)                            # dL/dy_hat                  (B,)
    g_w, g_b = X.T @ r, r.sum()                             # g = dL/dtheta (chain rule) (D,), ()
    w, b = w - lr * g_w, b - lr * g_b                       # theta <- theta - lr * g
    print(f"step {step}  L = {loss:.4f}  |g_w| = {np.linalg.norm(g_w):.4f}")
h = 1e-6                                                    # g_w[0]를 중앙차분으로 확인
L = lambda w0: np.mean((X @ np.r_[w0, w[1:]] + b - y) ** 2)
print("해석적 g_w[0] =", round(float(X[:, 0] @ (2 * (X @ w + b - y) / 8)), 6),
      " 수치미분 =", round((L(w[0] + h) - L(w[0] - h)) / (2 * h), 6))
```

```output
step 0  L = 2.6920  |g_w| = 2.6969
step 1  L = 2.0028  |g_w| = 2.1879
step 2  L = 1.5383  |g_w| = 1.8299
해석적 g_w[0] = -0.724451  수치미분 = -0.724451
```

NumPy 버전에서는 gradient를 매 step 새로 계산해 변수에 **대입**하므로 zero_grad가 필요 없다. PyTorch는 gradient를 `.grad`에 **더하므로**(8.4장의 `+=`, 17.2장의 누적) 비우는 줄이 따로 있다. 이 차이가 zero_grad 버그의 근원이다.

::quiz{id="update-line"}

## 역할의 구분

학습 루프에는 이름이 비슷해 섞이기 쉬운 세 쌍이 있다.

**forward와 backward.** forward는 값을 계산하며 그래프를 기록하고, backward는 그 그래프를 거꾸로 따라가며 gradient를 계산한다. backward의 비용은 forward의 작은 상수배이고, forward 때 저장한 중간값이 메모리의 대부분을 차지한다(8.4장). 그래서 평가처럼 gradient가 필요 없을 때 `no_grad`를 쓰면 메모리와 시간을 함께 아낀다.

**gradient 계산과 parameter update.** `backward()`는 파라미터를 바꾸지 않는다. 바꾸는 것은 `step()`이다. 학습이 안 될 때 이 둘을 나눠 확인한다. gradient가 0이거나 `None`이면 계산 쪽(그래프가 끊김, 파라미터가 optimizer에 등록되지 않음, 17.3장)의 문제이고, gradient는 정상인데 손실이 줄지 않으면 갱신 쪽(학습률, 스케줄, optimizer 설정)의 문제다.

**학습 손실과 보고용 metric.** 학습 손실은 **미분 가능하고 최적화하기 좋은** 함수여야 하고, metric은 **사람이 해석하고 과제의 목적에 맞는** 함수여야 한다. 둘이 같을 필요는 없다. 분류는 cross-entropy로 학습하고 정확도·F1으로 보고하며, 회귀는 표준화된 타깃의 MSE로 학습하고 원래 단위의 MAE로 보고하는 경우가 많다. 앞 셀에서 학습 손실은 MSE, 검증 metric은 MAE였다. 주의할 점은 학습 중 출력하는 batch 손실이 "현재 모델의 학습 데이터 손실"이 아니라는 것이다.

- epoch 동안의 batch 손실 평균은 계속 바뀐 **서로 다른 파라미터들**에서 잰 값의 평균이다.
- train 모드의 dropout·BatchNorm 때문에 같은 파라미터에서도 eval 모드와 값이 다르다.
- batch 크기가 다르면(마지막 batch) 단순 평균은 표본 수 가중 평균과 다르다.

::quiz{id="train-loss-vs-eval-loss"}

:::note[분야 예시]
신약 후보 분자의 결합 친화도 회귀에서, 학습 손실은 로그 단위 친화도(pKd)의 MSE로 두고, 보고는 실제 스크리닝에서 의미 있는 지표, 예를 들어 상위 1%로 고른 후보 중 실제 활성 분자의 비율(enrichment)로 하는 경우가 있다. MSE가 낮아져도 상위 순위의 정확도는 나아지지 않을 수 있다. 그래서 학습 손실의 감소와 별개로, 의사결정에 쓰일 metric을 검증 세트에서 함께 추적해 checkpoint를 고른다(17.6장).
:::

## 단위의 구분

**정의.** 학습의 진행을 세는 단위는 셋이다.

- **batch**(micro-batch): forward·backward 한 번에 들어가는 표본 묶음. 메모리가 크기를 제한한다.
- **step**: optimizer 갱신 한 번(`opt.step()` 한 번). gradient accumulation이 없으면 batch 하나가 step 하나다.
- **epoch**: 학습 데이터 전체를 한 번 훑는 것. epoch당 step 수는 데이터 크기·batch 크기·누적 횟수로 정해진다.

**gradient accumulation과 step의 의미.** micro-batch 크기 $b$, 누적 $k$회이면 유효 batch 크기는 $bk$이고 step은 $k$ batch마다 한 번이다. 10.4장에서 본 것처럼 micro-batch 손실을 평균한 뒤 누적하면 $k$로 나눠야 큰 batch의 gradient와 같다. 단위가 중요한 이유는 **학습률 스케줄, warm-up 길이, logging·checkpoint 주기가 모두 step으로 정의되기 때문**이다. 누적을 도입하면서 이 수들을 batch 단위로 그대로 두면 스케줄이 $k$배 빨리 진행한다.

```python {exec="browser" session="training-loop"}
N, micro_B, accum = 1000, 32, 4                             # 표본 수, micro-batch, 누적 횟수
micro_per_epoch = int(np.ceil(N / micro_B))                 # DataLoader가 내는 batch 수
steps_per_epoch = int(np.ceil(micro_per_epoch / accum))     # optimizer.step() 횟수
print(f"micro-batch {micro_per_epoch}개/epoch, optimizer step {steps_per_epoch}회/epoch, "
      f"유효 batch {micro_B * accum}")
total_steps = 10 * steps_per_epoch                          # 10 epoch 동안의 스케줄 길이
cosine = lambda k: 0.5 * 1e-3 * (1 + np.cos(np.pi * min(k, total_steps) / total_steps))
k_right = 5 * steps_per_epoch                               # 5 epoch 뒤, step 단위로 센 경우
k_wrong = 5 * micro_per_epoch                               # micro-batch마다 scheduler.step()
print(f"5 epoch 뒤 학습률  올바름 {cosine(k_right):.2e}   micro-batch마다 호출 {cosine(k_wrong):.2e}")
```

```output
micro-batch 32개/epoch, optimizer step 8회/epoch, 유효 batch 128
5 epoch 뒤 학습률  올바름 5.00e-04   micro-batch마다 호출 0.00e+00
```

10 epoch짜리 cosine 스케줄인데, scheduler를 micro-batch마다 부르면 2.5 epoch 만에 학습률이 0이 된다. 나머지 7.5 epoch은 사실상 학습하지 않는다. 손실 곡선이 일찍 평평해지는 것을 보고 "수렴했다"고 판단하기 쉬운 버그다.

::quiz{id="steps-per-epoch"}

::quiz{id="scheduler-per-micro-batch"}

## 실전 요소

**zero_grad의 위치.** `.grad` 버퍼의 규칙은 셋이다. backward는 더하고, step은 읽고, zero_grad는 비운다. 따라서 zero_grad는 이전 step이 버퍼를 읽은 뒤부터 이번 backward 전까지 어디에 있어도 되고, 그 밖에 두면 버그다.

```python {exec="python"}
def run(order, steps=60):
    torch.manual_seed(0)
    Xt = torch.randn(256, 3)                                # (N, D)
    yt = Xt @ torch.tensor([1.0, -2.0, 0.5]) + 0.3          # (N,)
    m = nn.Linear(3, 1)
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    for _ in range(steps):
        loss = nn.functional.mse_loss(m(Xt).squeeze(1), yt)
        if order == "올바름":
            o.zero_grad(); loss.backward(); o.step()
        elif order == "zero_grad 누락":
            loss.backward(); o.step()
        elif order == "backward 뒤 zero_grad":
            loss.backward(); o.zero_grad(); o.step()
    return loss.item(), [round(v, 3) for v in m.weight.detach().flatten().tolist()]

for order in ["올바름", "zero_grad 누락", "backward 뒤 zero_grad"]:
    final, wt = run(order)
    print(f"{order:20s} 최종 손실 {final:9.4f}  가중치 {wt}")
```

```output
올바름                  최종 손실    0.0001  가중치 [0.997, -1.993, 0.503]
zero_grad 누락         최종 손실    1.5136  가중치 [0.228, -1.659, 1.555]
backward 뒤 zero_grad 최종 손실    5.9050  가중치 [-0.443, -0.169, 0.189]
```

올바른 루프는 참값 $(1, -2, 0.5)$에 도달했다. zero_grad를 빠뜨리면 과거 gradient가 감쇠 없이 쌓여 최솟값을 지나쳐 진동하고(17.2장 실습과 같은 현상), backward와 step 사이에서 비우면 step이 빈 버퍼를 읽어 **초기값에서 한 걸음도 움직이지 않는다.** 집필 시점 PyTorch의 `zero_grad()`는 기본적으로 `.grad`를 0이 아니라 `None`으로 만들고, `step()`은 `.grad`가 `None`인 파라미터를 건너뛴다. 세 경우 모두 오류는 없다.

::quiz{id="zero-grad-position"}

**mixed precision.** 행렬 곱을 16비트 부동소수점(float16·bfloat16)으로 하면 GPU에서 속도가 빨라지고 활성값 메모리가 준다. 그러나 모든 연산을 16비트로 하면 작은 gradient가 0으로 사라지거나 합산 오차가 커진다. mixed precision은 연산마다 정밀도를 골라 쓴다(Micikevicius 외, 2018). 집필 시점 PyTorch에서는 두 도구를 쓴다.

- `torch.autocast(device_type, dtype)`: 블록 안의 연산을 연산 종류에 따라 16비트 또는 32비트로 실행한다. 파라미터 자체는 float32로 남는다. forward와 손실 계산만 감싸고, backward는 밖에서 한다.
- `torch.amp.GradScaler`: float16은 표현 범위가 좁아 작은 gradient가 0이 될 수 있으므로, 손실에 큰 수를 곱해 backward한 뒤 step 전에 되돌린다. bfloat16은 float32와 같은 지수 범위를 가져 보통 scaler가 필요 없다.

어느 연산이 어느 정밀도로 실행되는지는 device와 버전마다 다르므로, 결과의 dtype을 직접 확인한다.

**gradient clipping과 scheduler의 위치.** clipping(10.4장)은 계산된 gradient를 갱신 전에 줄이는 것이므로 backward 뒤, step 앞이다. GradScaler를 쓰면 `scaler.unscale_(optimizer)`로 scale을 되돌린 뒤 clipping한다. scheduler는 스케줄을 정의한 단위(step 또는 epoch)마다 **optimizer.step() 뒤에** 부른다.

```python {exec="python"}
import warnings

torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
o = torch.optim.SGD(m.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)
xb, yb = torch.randn(8, 4), torch.randn(8, 1)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):   # CPU에서는 bfloat16
    pred = m(xb)
    loss = nn.functional.mse_loss(pred.float(), yb)
print("autocast 안  pred:", pred.dtype, " loss:", loss.dtype, " 파라미터:", m[0].weight.dtype)
o.zero_grad()
loss.backward()
total = nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.5)   # backward 뒤, step 앞
after = torch.cat([p.grad.flatten() for p in m.parameters()]).norm()
print(f"clipping 전 |g| {total.item():.3f} -> 후 {after.item():.3f}")
o.step()
sched.step()                                                # optimizer.step() 뒤에
print("학습률:", o.param_groups[0]["lr"])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    o2 = torch.optim.SGD(m.parameters(), lr=0.1)
    torch.optim.lr_scheduler.StepLR(o2, step_size=1).step()   # step 전에 호출하면
print("경고:", any("before `optimizer.step()`" in str(c.message) for c in caught))
```

```output
autocast 안  pred: torch.bfloat16  loss: torch.float32  파라미터: torch.float32
clipping 전 |g| 3.791 -> 후 0.500
학습률: 0.05
경고: True
```

autocast 안에서 Linear의 출력은 bfloat16이지만 파라미터는 float32로 남았고, 손실은 float32로 계산했다. `clip_grad_norm_`은 clipping **전**의 전체 노름(3.791)을 돌려준다. 이 값을 매 step 기록하면 clipping이 얼마나 자주 발동하는지 알 수 있다(10.4장). optimizer.step() 전에 scheduler를 부르면 PyTorch가 경고를 낸다. 경고가 없는 경우(scheduler를 너무 자주 부르는 경우)가 더 위험하다.

::quiz{id="clip-position"}

한 step의 순서를 정리하면 다음과 같다. 대괄호는 쓸 때만 있는 줄이다.

```python
model.train()
for i, (xb, yb) in enumerate(loader):
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):   # [mixed precision]
        loss = loss_fn(model(xb), yb) / accum             # [누적이면 1/accum]
    scaler.scale(loss).backward()                         # scaler가 꺼져 있으면 loss.backward()와 같다
    if (i + 1) % accum == 0:
        scaler.unscale_(opt)                              # [clipping 전에 scale 되돌리기]
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)   # [clipping]
        scaler.step(opt); scaler.update()                 # opt.step()
        opt.zero_grad()
        sched.step()                                      # [step 단위 스케줄]
```

## end-to-end라는 말의 의미

**정의.** "end-to-end로 학습한다"는 입력을 받는 첫 구성 요소부터 손실까지의 **모든 학습 가능한 파라미터가 같은 손실의 gradient로 함께** 갱신된다는 뜻이다. 반대는 단계별 학습이다. 특성을 손으로 설계하거나, 사전학습한 인코더를 고정하거나, 구성 요소를 각자의 손실로 따로 학습한 뒤 이어 붙이는 방식이다.

**연구에서의 의미.** end-to-end 학습은 앞 단계가 최종 목적에 맞게 조정된다는 장점이 있지만, 데이터가 많이 필요하고, 중간 표현의 의미가 보장되지 않는다. 단계별 방식은 각 단계를 따로 검증할 수 있고 물리적으로 의미 있는 중간량을 유지할 수 있다. 과학 문제에서는 둘을 섞는 경우가 많으므로, 논문과 보고서에서 **실제로 공동 최적화된 부분이 어디까지인지** 밝혀야 한다. 다음을 확인하면 된다.

- 어떤 파라미터가 optimizer에 들어갔는가(`requires_grad=False`로 고정한 부분, optimizer에 넘기지 않은 부분)
- 입력 정규화 통계, 토크나이저, 특성 추출처럼 학습 밖에서 정해진 전처리는 무엇인가
- 손실이 여러 개이면, 각 손실의 gradient가 어느 파라미터까지 흐르는가(`detach`로 끊은 곳)

::quiz{id="end-to-end-meaning"}

:::note[분야 예시]
기상 레이더 영상으로 강수량을 예측하는 문제에서, 전통적인 방법은 반사도를 경험식(Z–R 관계)으로 강수 강도로 바꾼 뒤 외삽한다. 신경망을 그 뒤에만 붙이면 경험식의 오차는 학습으로 고쳐지지 않는다. 반사도 영상에서 강수량까지를 하나의 손실로 학습하면 변환까지 데이터에 맞춰지지만, 중간 표현이 더는 물리량이 아니게 된다. 어느 쪽을 택했는지가 결과 해석과 외삽 성능 평가에 영향을 준다.
:::

## 실습

프레임워크 없이 학습 루프를 처음부터 작성하고, 각 줄을 수식에 대응시키는 주석을 단다. 코드는 `labs/deep-learning-pytorch/training-loop/loop_from_scratch.py`다. NumPy만으로 2층 tanh MLP의 forward, 손실, 손으로 쓴 backward, Adam 갱신(10.4장), epoch마다 섞는 mini-batch 순서(17.4장)를 구현하고 다음을 확인한다.

1. 32개 표본에 3000 step 학습해 손실이 줄어드는지 본다.
2. **batch 하나에 과적합되는지** 본다. 표본 8개짜리 batch 하나만 반복해 학습하면 손실이 거의 0이 되어야 한다. 이것이 안 되면 모델·손실·gradient·갱신 중 어딘가에 버그가 있다. 데이터 전체로 학습하기 전에 하는 가장 싼 점검이다.
3. 같은 초기값·같은 batch 순서·float64로 PyTorch(`nn.Sequential` + `torch.optim.Adam`) 루프를 돌려, 매 step의 손실이 NumPy 루프와 일치하는지 본다.

집필 시점의 실행 결과는 다음과 같다(PyTorch 2.15, CPU).

```output
[NumPy] 표본 32개, batch 16, Adam lr=0.01, 3000 step
  step    0  batch 손실 1.202132
  step  100  batch 손실 0.093872
  step  500  batch 손실 0.067402
  step 1000  batch 손실 0.049800
  step 2000  batch 손실 0.011657
  step 2999  batch 손실 0.004709
  학습 데이터 전체 MSE 6.60e-03  (잡음 분산 1.0e-02보다 작으면 잡음까지 외운 것)
[과적합 점검] batch 하나(표본 8개)를 2000 step: 손실 1.280 -> 4.3e-08
[PyTorch] 같은 초기값·같은 batch 순서·float64
  step    0  |손실 차이| 0.0e+00
  step  100  |손실 차이| 2.8e-17
  step 1000  |손실 차이| 1.4e-17
  step 2999  |손실 차이| 5.2e-18
  최종 W1 최대 차이 1.7e-14
```

결과를 읽는 법은 세 가지다.

- **batch 하나에는 완전히 과적합된다**($4.3 \times 10^{-8}$). 루프의 모든 줄이 맞게 동작한다는 증거다.
- **전체 학습 MSE(0.0066)가 잡음 분산(0.01)보다 작다.** 32개 표본에 파라미터 97개인 모델이 잡음까지 일부 외웠다는 뜻이다. 학습 손실만으로는 이것이 좋은지 알 수 없고, 검증 데이터가 필요하다(17.6장).
- **NumPy 루프와 PyTorch 루프가 3000 step 동안 반올림 수준($10^{-17}$)으로 일치한다.** 8.4장 실습에서는 학습률 1.0의 SGD가 작은 차이를 증폭해 궤적이 갈라졌지만, 이번 설정에서는 갈라지지 않았다. 궤적의 일치는 이런 비교에서 보너스일 뿐이고, 구현의 검증은 같은 파라미터에서의 gradient 비교로 한다는 원칙은 그대로다.

이어서 두 연습문제를 푼다. 첫째는 `.grad` 버퍼의 규칙으로 루프의 순서 버그를 예측하는 시뮬레이터이고, 둘째는 gradient accumulation이 큰 batch와 같은 학습이 되도록 루프를 쓰는 것이다.

::exercise{id="grad-buffer-sim"}

::exercise{id="accumulation-loop"}

## 실패 사례

- **zero_grad 위치 오류로 gradient가 누적되는 것.** 빠뜨리면 과거 gradient가 쌓여 진동하고, backward와 step 사이에 두면 학습이 멈춘다. 둘 다 오류가 없다. 처음 몇 step 동안 파라미터의 변화량과 gradient 노름을 기록하면 바로 드러난다. 누적을 쓸 때는 zero_grad를 누적 주기에 맞춰 step 직후에만 둔다.
- **scheduler를 잘못된 주기로 호출하는 것.** step 단위로 정의한 스케줄을 micro-batch마다, 또는 epoch 단위 스케줄을 step마다 부르면 스케줄이 몇 배 빨리 끝난다. 학습률을 매 step 기록해 계획한 곡선과 겹쳐 보는 것이 가장 확실한 확인 방법이다.
- **batch 손실의 이동 평균을 모델의 학습 손실로 보고하는 것.** 서로 다른 파라미터에서 잰 값의 평균이고 train 모드의 값이다. 비교·보고에는 고정한 파라미터와 eval 모드로 다시 계산한 값을 쓴다.
- **과적합 점검 없이 전체 학습부터 시작하는 것.** batch 하나에 과적합하지 못하는 루프는 전체 데이터에서도 실패한다. 큰 실험을 돌리기 전에 몇 분 안에 끝나는 이 점검을 먼저 한다(17.9장).

## 통과 기준

- 실습 코드로 batch 하나(표본 8개)에 손실 $10^{-6}$ 이하까지 과적합하고, 같은 루프를 PyTorch로 옮겨 같은 초기값·같은 batch 순서에서 매 step 손실이 $10^{-10}$ 이내로 일치함을 보인다.
- 학습 루프의 모든 줄을 가리키며 대응하는 수식, 하는 일, 그 줄이 빠지거나 순서가 바뀌었을 때의 증상을 설명한다.
- 연습문제 두 개(`grad-buffer-sim`, `accumulation-loop`)의 테스트를 통과한다.

## 참고 문헌

- Micikevicius 외 (2018). Mixed precision training. *ICLR 2018*. — float16 학습에서 float32 가중치 사본과 loss scaling
- Kingma, Ba (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980. — 실습의 Adam 갱신식
