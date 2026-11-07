---
id: deep-learning-pytorch/dashboards
title: "학습 모니터링과 대시보드"
title_en: "Monitoring and Dashboards"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/training-loop]
---

# 학습 모니터링과 대시보드

학습은 몇 시간에서 며칠이 걸리고, 그동안 연구자가 볼 수 있는 것은 기록된 숫자뿐이다. 무엇을 기록했는지가 무엇을 알 수 있는지를 정한다. 손실만 기록한 학습이 NaN으로 끝나면 "언젠가 발산했다"는 것밖에 모르지만, gradient norm과 학습률을 함께 기록했다면 발산이 언제, 어떤 신호와 함께 시작되었는지 볼 수 있다.

대시보드는 이 기록을 그래프로 보여 주는 도구다. 편리한 만큼 착시도 만든다. smoothing 슬라이더는 spike를 지우고 상승을 늦게 보여 주며, 화면에 그려지는 점은 기록된 점의 일부일 수 있고, "train loss"라는 이름의 곡선이 무엇의 평균인지는 기록한 코드만 안다. 이 장은 무엇을 기록할지 정하고, 대시보드를 운영하고, 곡선에서 다음 실험의 방향을 읽는 방법을 다루면서, 화면의 곡선과 원본 숫자를 대조하는 습관을 함께 익힌다. 기록의 형식과 로깅 라이브러리는 6.5장, 여러 실행의 비교와 결과 관리는 22.1장에서 다룬다.

## 학습 목표

- 무엇을 기록할지 결정한다
- GUI 대시보드를 운영한다
- 학습 곡선을 해석한다

## 무엇을 기록하는가

손실 곡선은 결과다. 학습이 잘못될 때 원인은 손실보다 먼저 다른 숫자에 나타난다. 기록할 것을 고르는 기준은 "이 숫자가 이상하면 무엇을 의심하게 되는가"다.

| 종류 | 예 | 이상하면 의심하는 것 |
|---|---|---|
| 손실·metric | train 손실, 검증 손실, 검증 metric | 과적합·과소적합, 평가 버그 |
| 최적화 상태 | 실제 학습률, global gradient norm, 층별 gradient norm | scheduler 설정, 발산, 기울기 소실 |
| 파라미터 | 파라미터 노름, 업데이트 비율 $\lVert\Delta\theta\rVert / \lVert\theta\rVert$ | 학습률 크기, weight decay |
| 활성값 | 층별 평균·표준편차, 포화(0 또는 ±1) 비율 | 초기화, 정규화 누락(17.9장) |
| 처리량 | step 시간, 데이터 대기 시간, 초당 표본 수, GPU 메모리 | 데이터 로더 병목, 메모리 누수(22.4장) |
| 예측 표본 | 고정한 검증 입력 몇 개의 예측 그림 | metric이 놓치는 질적 실패 |
| 설정 | 하이퍼파라미터, 코드 버전 | 실행 사이의 차이 |

**업데이트 비율**은 한 step에서 파라미터가 자기 크기에 비해 얼마나 움직였는지다. 학습률·gradient 크기·파라미터 스케일이 한 숫자에 합쳐져 있어서, 학습률이 문제에 비해 큰지 작은지를 손실보다 직접 보여 준다. 적절한 값은 모델과 optimizer에 따라 다르므로, 잘 학습되는 실행을 기준선으로 삼고 그와 비교한다. 같은 작은 MLP를 학습률 0.05와 0.5로 학습하며 이 값들을 찍어 본다.

```python {exec="python"}
import torch
from torch import nn

def train_and_log(lr, steps=60, every=15):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x = torch.randn(256, 8)                                        # (N, 8)
    y = x[:, :1] * 2 - x[:, 1:2] + 0.1 * torch.randn(256, 1)       # (N, 1)
    for step in range(1, steps + 1):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        before = torch.cat([p.detach().flatten() for p in model.parameters()])
        opt.step()
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        if step % every == 0 or not torch.isfinite(loss):
            print(f"lr {lr:<4}  step {step:3d}  loss {loss.item():10.4g}  |g| {grads.norm():9.3g}  "
                  f"|θ| {after.norm():9.3g}  |Δθ|/|θ| {(after - before).norm() / before.norm():9.2e}")
        if not torch.isfinite(loss):
            break

train_and_log(0.05)
train_and_log(0.5, every=2)
```

```output
lr 0.05  step  15  loss     0.9147  |g|      2.16  |θ|      4.02  |Δθ|/|θ|  2.72e-02
lr 0.05  step  30  loss    0.09494  |g|     0.261  |θ|      4.28  |Δθ|/|θ|  3.06e-03
lr 0.05  step  45  loss    0.06656  |g|     0.149  |θ|      4.28  |Δθ|/|θ|  1.74e-03
lr 0.05  step  60  loss    0.05426  |g|     0.108  |θ|      4.28  |Δθ|/|θ|  1.27e-03
lr 0.5   step   2  loss       3.62  |g|      3.87  |θ|      4.25  |Δθ|/|θ|  5.31e-01
lr 0.5   step   4  loss      71.67  |g|      67.2  |θ|      29.6  |Δθ|/|θ|  5.92e+00
lr 0.5   step   6  loss   1.38e+08  |g|  4.31e+06  |θ|  2.15e+06  |Δθ|/|θ|  6.32e+03
lr 0.5   step   8  loss        inf  |g|       nan  |θ|       nan  |Δθ|/|θ|       nan
```

학습률 0.05에서는 gradient norm이 줄고 파라미터 노름이 안정되며 업데이트 비율이 $10^{-3}$ 근처로 내려간다. 학습률 0.5에서는 step 2에 이미 업데이트 비율이 0.53이다. 한 step에 파라미터가 자기 크기의 절반 넘게 움직인 것이다. step 4에는 비율이 1을 넘고, 손실이 inf가 되는 것은 step 8이다. 손실만 기록했다면 step 8에서야 알았을 일을 step 2에서 볼 수 있다.

::quiz{id="what-to-log-before-divergence"}

**기록 주기.** 모든 것을 매 step 기록할 필요는 없다. 비용과 쓸모로 나눈다.

- **매 step(또는 수십 step마다)**: 손실, 학습률, global gradient norm. 스칼라 몇 개라 비용이 거의 없고, 발산 직전의 변화를 잡으려면 촘촘해야 한다.
- **수백–수천 step마다**: 층별 통계, 파라미터·gradient 히스토그램, 처리량. 계산이나 저장이 조금 더 든다.
- **검증 주기마다**: 검증 손실·metric, 예측 표본 그림.

GPU에서 `loss.item()`처럼 텐서를 Python 숫자로 바꾸는 호출은 GPU 작업이 끝날 때까지 기다리게 만든다. 매 step 여러 번 호출하면 학습이 느려질 수 있으므로, 기록할 값은 텐서로 모았다가 기록 주기에 한 번 꺼낸다(22.4장).

## TensorBoard

TensorBoard는 학습 중에 기록한 값을 이벤트 파일로 쓰고, 웹 서버가 그 파일을 읽어 브라우저에 그래프로 보여 주는 도구다. 원래 TensorFlow와 함께 개발되었지만 PyTorch에서도 `torch.utils.tensorboard.SummaryWriter`로 쓴다. 이 책의 실습 공통 환경에는 들어 있지 않으므로, 필요할 때 `uv run --with tensorboard`로 그 실행에만 더한다(집필 시점 예시. tensorboard 2.21.0, PyTorch 2.15에서 확인).

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="outputs/tb/run1")      # 실행마다 다른 하위 디렉터리
for step in range(num_steps):
    ...
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
    if step % 1000 == 0:
        writer.add_histogram("weights/fc1", model.fc1.weight.detach(), step)
writer.add_text("notes", "baseline, 시드 0", 0)
writer.add_hparams({"lr": 1e-3, "batch_size": 32}, {"hparam/val_loss": best_val})
writer.close()
```

기록할 수 있는 주요 형식은 다음과 같다.

| 형식 | 메서드 | 용도 |
|---|---|---|
| scalar | `add_scalar` | 손실, 학습률, gradient norm 같은 시계열 |
| histogram | `add_histogram` | 가중치·gradient·활성값 분포의 시간 변화 |
| image | `add_image`, `add_figure` | 예측 그림, matplotlib 그림 |
| text | `add_text` | 실행 메모, 예측 문자열 |
| hparams | `add_hparams` | 하이퍼파라미터와 최종 metric을 표로 비교 |

태그에 `train/`, `val/`처럼 접두사를 붙이면 화면에서 묶여 표시된다. **실행마다 다른 하위 디렉터리**에 쓰고 상위 디렉터리를 `--logdir`로 주면, 여러 실행의 곡선이 한 그래프에 겹쳐 보인다. 디렉터리 이름에 실행 id를 쓰면(6.1장) 대시보드의 곡선과 결과 폴더를 바로 연결할 수 있다.

**원격 서버에서 보기.** 학습은 서버에서 돌고 브라우저는 노트북에 있다. 안전한 방법은 서버에서 TensorBoard를 기본값(자기 자신, 즉 localhost에서만 접속을 받음)으로 띄우고, SSH 포트 포워딩(2.3장)으로 노트북의 포트를 서버의 포트에 잇는 것이다.

```bash
# 서버에서 (tmux 같은 세션 안에서 띄워 두면 연결이 끊겨도 유지된다, 2.2장)
uv run --project labs --group torch --with tensorboard tensorboard --logdir outputs/tb --port 6006

# 노트북에서
ssh -L 6006:localhost:6006 user@server
# 브라우저에서 http://localhost:6006
```

공유 서버에서는 다른 사용자가 이미 6006번을 쓰고 있을 수 있다. 다른 번호를 고르거나 `--port 0`으로 운영체제가 빈 포트를 고르게 한다(집필 시점 `tensorboard --help`의 설명). `--bind_all` 옵션은 모든 네트워크 인터페이스에서 접속을 받게 하는데, TensorBoard에는 기본 인증이 없으므로 대시보드와 그 안의 정보가 같은 네트워크의 누구에게나 열린다. 포트 포워딩으로 충분하면 쓰지 않는다.

::quiz{id="tensorboard-remote-access"}

**화면이 기본으로 하는 일.** TensorBoard 웹 화면에는 기본값이 몇 가지 있다. 집필 시점(2.21.0) 웹 화면의 코드에서 확인한 바로는 scalar 그래프의 smoothing 가중치 기본값이 0.6이고, y축 범위를 정할 때 이상치를 무시하는 설정이 켜져 있다. 또 `--help`는 기록된 값이 많으면 TensorBoard가 플러그인별로 표본을 **무작위로 줄인다**고 설명한다. 이 세 가지가 모두 "화면의 곡선 ≠ 기록된 원본"을 만든다. 마지막 절에서 그 영향을 확인한다.

## 다른 도구

TensorBoard 말고도 학습 기록과 대시보드를 제공하는 도구가 여럿 있다. 아래는 역할을 일반적으로 정리한 것이다(집필 시점 예시. 이 책의 실습 환경에는 설치되어 있지 않으며, 기능·요금·데이터 정책은 각 도구의 문서에서 확인한다).

- **Weights & Biases**: 클라우드 서비스를 중심으로 한 실험 기록·대시보드 도구. 실행 비교, 설정 기록, 팀 공유, 하이퍼파라미터 탐색 기능을 함께 제공한다.
- **MLflow**: 실험 기록(parameter·metric·결과 파일)과 모델 관리를 위한 오픈소스 플랫폼. 자체 서버를 두고 쓸 수 있다.
- **Aim**: 오픈소스 실험 기록·시각화 도구. 로컬이나 자체 서버에서 많은 실행을 비교하는 데 초점을 둔다.

도구를 고를 때 갈리는 것은 기능보다 **데이터가 어디에 저장되는가**다.

| | 로컬·자체 서버 (TensorBoard, 자체 호스팅 MLflow·Aim) | 클라우드 서비스 |
|---|---|---|
| 데이터 위치 | 내 서버의 파일 | 서비스 업체의 서버 |
| 공유 | 포트 포워딩이나 서버 운영이 필요 | 링크로 공유 |
| 보안·규정 | 데이터가 밖으로 나가지 않음 | 기관의 데이터 정책, 계정 관리 확인 필요 |
| 오프라인 | 네트워크 없이 동작 | 네트워크가 끊기면 지연 업로드 또는 기록 누락 가능 |
| 운영 | 디스크·서버 관리 직접 | 업체가 관리, 서비스 정책 변경에 영향받음 |

환자 데이터, 공개 전 실험 데이터, 산업 협력 과제처럼 외부 반출이 제한된 데이터를 다룬다면 예측 표본 그림이나 텍스트가 클라우드로 올라가는 것도 반출일 수 있다. 어느 도구를 쓰든 **원본 기록을 로컬 파일로도 남기는** 습관을 들인다. 대시보드 서비스가 바뀌거나 사라져도 결과는 남아야 한다(22.1장).

## 학습 곡선 해석

곡선을 읽는 목적은 **다음 실험을 정하는 것**이다. 그래서 곡선의 모양을 진단으로, 진단을 처방으로 바꾸는 표가 필요하다. 네 가지 전형적인 모양을 합성한 곡선으로 만들고, 사람이 눈으로 보는 특징을 숫자로 바꾸어 본다. 마지막 train 손실, 검증 손실이 가장 낮은 epoch, 마지막 epoch의 gap(검증 − train), 마지막 10 epoch의 검증 기울기, 검증 손실의 epoch 간 흔들림이다.

```python {exec="browser" session="dash"}
import numpy as np

ep = np.arange(1, 51)
noise = lambda s: 0.01 * np.random.default_rng(s).normal(size=50)
curves = {
    "과적합":   (1.0 * np.exp(-ep / 8) + 0.05,       0.9 * np.exp(-ep / 8) + 0.25 + 0.004 * ep),
    "과소적합": (0.6 + 0.2 * np.exp(-ep / 5),       0.62 + 0.2 * np.exp(-ep / 5)),
    "학습률 큼": (0.4 + 0.3 * np.abs(np.sin(ep)),   0.45 + 0.3 * np.abs(np.sin(ep + 1))),
    "학습률 작음": (1.0 - 0.006 * ep,               1.02 - 0.006 * ep),
}
print("모양         마지막 train  val 최저 epoch  마지막 gap  마지막 10 epoch val 기울기  val 흔들림")
for i, (name, (tr, va)) in enumerate(curves.items()):
    tr, va = tr + noise(i), va + noise(i + 10)                           # (50,), (50,)
    slope = np.polyfit(ep[-10:], va[-10:], 1)[0]
    wiggle = np.std(np.diff(va))
    print(f"{name:10s} {tr[-1]:10.3f}  {va.argmin() + 1:10d}  {va[-1] - tr[-1]:12.3f}  {slope:20.4f}  {wiggle:12.3f}")
```

```output
모양         마지막 train  val 최저 epoch  마지막 gap  마지막 10 epoch val 기울기  val 흔들림
과적합             0.065          26         0.385                0.0039         0.025
과소적합            0.609          25         0.004               -0.0009         0.013
학습률 큼           0.490          21         0.167               -0.0054         0.155
학습률 작음          0.711          48         0.023               -0.0040         0.016
```

| 모양 | 숫자에서 보이는 것 | 다음 실험의 방향 |
|---|---|---|
| 과적합 | train 낮음, gap 큼, 검증 기울기 양수(상승), 검증 최저가 중간 | regularization(weight decay·dropout·증강) 강화, 모델 축소, 데이터 확대. 그동안 best checkpoint 사용 |
| 과소적합 | train도 높은 곳에서 평평, gap 작음 | 모델 용량 확대, 특징·입력 표현 개선, 학습률·학습 길이 점검 |
| 학습률 큼 | 검증 흔들림이 다른 모양보다 여섯 배 이상 큼 | 학습률 축소, warm-up, gradient clipping(10.4장) |
| 학습률 작음 | 끝까지 꾸준히 내려가는 중(최저가 마지막 근처), 기울기 음수 | 학습률 확대 또는 학습 연장 |

과소적합과 학습률 작음은 둘 다 train 손실이 높다는 점에서 비슷해 보인다. 차이는 **아직 내려가고 있는가**다. 마지막 구간의 기울기가 음수로 뚜렷하면 학습이 덜 된 것이고, 0에 가까우면 모델이나 입력이 한계에 이른 것이다. 그래서 곡선의 마지막 값만이 아니라 기울기를 본다.

이 밖에 자주 보이는 모양이 있다.

- **갑자기 NaN·inf**: 발산. gradient norm과 업데이트 비율 기록에서 원인을 찾는다(17.9장).
- **계단처럼 뚝 떨어지는 손실**: 학습률을 단계적으로 줄이는 스케줄의 효과인 경우가 많다. 학습률 곡선과 겹쳐 본다.
- **처음부터 평평한 손실**: 학습이 전혀 안 되는 경우. 초기 손실이 이론값(예: $C$개 class 분류에서 $\ln C$)과 같은지, gradient가 0이 아닌지 확인한다(17.9장).
- **검증 손실이 train보다 낮음**: 버그일 수도 있지만, 집계 방식과 dropout만으로도 흔히 생긴다. 마지막 절의 예를 본다.
- **검증 손실은 오르는데 검증 정확도도 오름**: 분류에서 모델이 맞히는 표본은 늘지만 틀리는 표본에 대한 확신이 과도해지는 경우다. 손실과 metric 중 무엇으로 best를 고를지 미리 정해 둔다(17.6장).

::quiz{id="overfitting-pattern"}

:::note[분야 예시]
기상 예측 모델은 학습 손실과 함께 예측 길이(lead time)별 오차를 검증 metric으로 기록하는 경우가 많다. 1일 예측의 오차는 계속 줄어드는데 10일 예측의 오차가 어느 시점부터 오르면, 평균 손실 곡선 하나로는 보이지 않는 과적합이다. 분자 물성 예측에서는 화학적으로 다른 골격(scaffold)으로 나눈 검증 세트의 곡선을 따로 그리면, 무작위 분할 검증 곡선이 숨기는 일반화 실패가 드러난다(15.4장). 무엇을 기록할지는 연구 질문이 정한다.
:::

## 대시보드와 원본의 일치

대시보드의 곡선은 기록된 숫자를 가공한 것이다. 결론을 내리기 전에 가공이 무엇을 바꾸었는지 확인한다. 세 가지를 차례로 본다.

**smoothing.** 잡음이 많은 손실 곡선을 매끄럽게 보이려고 대시보드는 지수이동평균(exponential moving average, EMA)을 쓴다. 상태 $s_0 = 0$에서 시작해

$$
s_k = w\, s_{k-1} + (1 - w)\, x_k, \qquad \tilde{x}_k = \frac{s_k}{1 - w^k}
$$

를 그린다. $w$가 smoothing 가중치이고, $1 - w^k$로 나누는 것은 0에서 출발한 치우침을 지우는 보정(debias)이다. EMA는 대략 최근 $1/(1-w)$개의 값을 평균한 것처럼 행동한다. $w = 0.9$이면 약 10 step, $w = 0.99$이면 약 100 step이다. 집필 시점 TensorBoard 웹 화면의 smoothing이 이 보정한 EMA다(웹 화면 코드에서 확인).

이 평균이 무엇을 숨기는지 합성 곡선으로 본다. 손실은 내려가다가 step 800부터 천천히 오르고, step 600에 5 step짜리 spike가 있다.

```python {exec="browser" session="dash"}
rng = np.random.default_rng(0)
steps = np.arange(1, 1001)
base = 2.0 * np.exp(-steps / 150) + 0.3 + 0.0008 * np.maximum(steps - 800, 0)   # 800 step부터 상승
loss = base + 0.08 * rng.normal(size=1000)                               # (1000,) 기록된 원본
loss[600:605] += 3.0                                                     # 5 step짜리 spike

def ema(x, weight, debias):
    s, out = 0.0, np.empty_like(x)
    for i, v in enumerate(x):
        s = weight * s + (1 - weight) * v
        out[i] = s / (1 - weight ** (i + 1)) if debias else s
    return out

print(f"원본      : 상승 구간(900–1000)에서 기저 곡선보다 평균 {np.mean(base[899:] - loss[899:]):.3f} 낮게 보임")
for w in (0.9, 0.99):
    sm = ema(loss, w, debias=True)
    print(f"EMA {w:<4} : 상승 구간(900–1000)에서 기저 곡선보다 평균 {np.mean(base[899:] - sm[899:]):.3f} 낮게 보임, "
          f"spike 최대 {loss[595:615].max():.2f} → {sm[595:615].max():.2f}")
print(f"보정 없는 EMA의 첫 값 {ema(loss, 0.99, debias=False)[0]:.3f}, 원본 첫 값 {loss[0]:.3f}")
```

```output
원본      : 상승 구간(900–1000)에서 기저 곡선보다 평균 0.021 낮게 보임
EMA 0.9  : 상승 구간(900–1000)에서 기저 곡선보다 평균 0.028 낮게 보임, spike 최대 3.52 → 1.55
EMA 0.99 : 상승 구간(900–1000)에서 기저 곡선보다 평균 0.058 낮게 보임, spike 최대 3.52 → 0.54
보정 없는 EMA의 첫 값 0.023, 원본 첫 값 2.297
```

- **spike가 사라진다.** 3.52까지 튄 손실이 $w = 0.99$에서는 0.54로 보인다. 데이터 한 batch가 깨졌거나 수치 문제가 잠깐 생긴 신호를 놓친다.
- **추세를 늦게 보여 준다.** 손실이 오르는 구간에서 smoothed 곡선은 실제보다 낮게, 즉 아직 덜 나빠진 것처럼 보인다. 원본도 이 구간의 잡음 때문에 기저 곡선보다 평균 0.021 낮지만, $w = 0.99$의 smoothed 곡선은 그보다 약 0.04 더 낮다. 이 차이가 smoothing의 지연이다. 과적합이 시작된 시점을 늦게 판단하게 된다.
- **보정이 없으면 시작이 0 쪽으로 끌린다.** 보정 없는 EMA의 첫 값은 0.023으로, 실제 손실 2.297의 1%다.

::widget{name="smoothing-illusion"}

::quiz{id="ema-without-debias"}

**표본 줄이기.** 기록된 점이 많으면 TensorBoard는 태그마다 일부만 남겨 보여 준다. 실습의 `tensorboard_demo.py`는 20000 step의 손실을 기록하고 step 12345에 spike를 하나 넣은 뒤, TensorBoard의 이벤트 읽기 도구(`EventAccumulator`)로 다시 읽는다. 집필 시점(tensorboard 2.21.0)의 결과는 다음과 같다.

```output
기본 한도: {'scalars': 10000, 'histograms': 1}
기본 설정: 읽은 점 10000개  최댓값 0.980  step 12345 포함 False
전부 읽기: 읽은 점 20000개  최댓값 5.008  step 12345 포함 True
태그: ['train/loss', 'train/lr'] ['weights/linear']
```

기본 설정은 scalar를 태그마다 10000개까지만 무작위로 남겼고, spike가 빠져 최댓값이 0.980이 되었다. 같은 로그를 TensorBoard 웹 서버로 열고 scalar 데이터 요청을 직접 보내 보니, 20000개 중 1000개가 돌아왔다(이 실행에서는 spike가 우연히 포함되었다). 남는 점은 무작위이므로, **드문 사건은 화면에 나올 수도 안 나올 수도 있다.** 드문 사건을 확실히 보려면 사건 자체를 따로 기록한다. 예를 들어 손실이 직전 평균의 몇 배를 넘거나 NaN이 된 step 번호를 별도 태그나 로그 파일에 남긴다(17.9장).

**기록 방식.** 같은 "train loss"라도 무엇을 기록했는지에 따라 다른 숫자다. 선형 회귀 한 epoch(batch 100개)를 학습하며 세 가지 방식으로 기록해 본다.

```python {exec="browser" session="dash"}
rng = np.random.default_rng(1)
X = rng.normal(size=(2000, 5))                                           # (N, D)
y = X @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]) + 0.5 * rng.normal(size=2000)
w, lr, batch_losses = np.zeros(5), 0.02, []
for i in range(0, 2000, 20):                                             # 한 epoch, batch 100개
    Xb, yb = X[i:i + 20], y[i:i + 20]
    r = Xb @ w - yb
    batch_losses.append(np.mean(r ** 2))                                 # 업데이트 전 batch 손실
    w -= lr * 2 * Xb.T @ r / 20
print(f"epoch 동안 batch 손실의 평균(대시보드)  {np.mean(batch_losses):.3f}")
print(f"epoch 끝 모델로 다시 계산한 train 손실  {np.mean((X @ w - y) ** 2):.3f}")
print(f"100 step마다 찍은 batch 손실 하나        {batch_losses[-1]:.3f}  (마지막 batch)")
print(f"잡음 분산(도달 가능한 최소 손실)        {0.5 ** 2:.3f}")
```

```output
epoch 동안 batch 손실의 평균(대시보드)  1.053
epoch 끝 모델로 다시 계산한 train 손실  0.242
100 step마다 찍은 batch 손실 하나        0.295  (마지막 batch)
잡음 분산(도달 가능한 최소 손실)        0.250
```

epoch 동안 batch 손실을 평균한 값(1.053)은 epoch 초반의 나쁜 모델과 후반의 좋은 모델을 섞은 것이다. epoch 끝 모델로 다시 계산한 손실은 0.242로, 잡음 분산 0.25에 거의 도달했다. 100 step마다 찍은 batch 손실 하나(0.295)는 그 batch 20개의 잡음을 그대로 담는다. 대시보드의 train 손실이 검증 손실보다 높게 보이는 흔한 이유가 이것이다. 신경망이라면 학습 모드의 dropout이 train 손실을 더 높인다.

::quiz{id="dashboard-train-loss"}

**원본과의 대조.** 마지막으로, 대시보드에 찍힌 검증 metric은 학습 코드 안의 평가 함수가 계산한 것이다. 논문에 쓸 숫자는 checkpoint를 독립 평가 스크립트로 다시 계산해 얻고(17.6장), 대시보드의 값과 같은지 확인한다. 다르다면 평가 모드, 집계, 전처리 중 하나가 두 코드에서 다르다는 뜻이다. 대시보드는 학습을 **관찰**하는 도구이고, 결과를 **보고**하는 원천은 원본 기록과 평가 스크립트다.

## 실습

원격 서버의 학습을 TensorBoard로 실시간 관찰하고, 이상 패턴을 하나 찾아 원인을 분석한다. 코드는 `labs/deep-learning-pytorch/dashboards/`에 있다.

1. `tensorboard_demo.py`를 실행하고 TensorBoard를 띄워, smoothing 슬라이더를 0, 0.6, 0.99로 바꾸며 step 12345의 spike가 어떻게 보이는지 확인한다. 위의 출력처럼 원본을 다시 읽어 화면과 비교한다.

   ```bash
   uv run --project labs --group torch --with tensorboard python labs/deep-learning-pytorch/dashboards/tensorboard_demo.py
   uv run --project labs --group torch --with tensorboard tensorboard --logdir labs/deep-learning-pytorch/dashboards/outputs/tb --port 6006
   ```

2. 17.6장이나 17.7장 실습의 학습 스크립트에 `SummaryWriter`를 붙여 손실·학습률·global gradient norm·업데이트 비율을 step마다, 검증 metric을 검증마다 기록한다. 원격 서버에서 실행하고, 노트북에서 SSH 포트 포워딩으로 학습 중의 곡선을 본다.
3. 학습률을 10배 키운 실행과 weight decay를 끈 실행을 같은 `--logdir` 아래 추가로 띄우고, 세 실행의 곡선을 겹쳐 본다. 이상 패턴(발산 전 업데이트 비율의 급증, 과적합의 시작 등)을 하나 골라, **어느 기록에서 처음 보였는지**, smoothing을 끄면 무엇이 달라지는지, 원인과 다음 실험을 한 문단으로 쓴다.
4. 학습이 끝나면 대시보드의 마지막 검증 metric과 독립 평가 스크립트의 결과를 대조한다.

이어서 연습문제로 대시보드의 smoothing을 직접 구현한다.

::exercise{id="debiased-ema"}

## 실패 사례

- **smoothing된 곡선으로 결론을 내리는 것.** 두 실행의 차이가 smoothing 가중치 0.99의 곡선에서만 보인다면, 그 차이는 원본의 잡음 범위 안에 있을 수 있다. 반대로 spike나 발산의 시작은 smoothing에 묻힌다. 결론을 내리기 전에 smoothing을 0으로 두고 원본을 본다. 여러 시드의 변동 폭과 비교하는 것이 그다음이다(22.2장).
- **대시보드 수치와 평가 스크립트 결과가 다른 것을 놓치는 것.** 학습 중 검증 함수가 `eval()`을 빠뜨렸거나, batch 평균의 평균을 쓰거나, 전처리 통계가 다르면 대시보드의 숫자는 평가 스크립트와 어긋난다. 대시보드의 best를 그대로 논문에 옮기면 이 차이가 결과에 들어간다.
- **x축이 다른 곡선을 겹쳐 비교하는 것.** batch 크기가 다른 두 실행을 step 축으로 겹치면, 같은 step에서 본 데이터 양이 다르다. 본 표본 수나 epoch, 또는 같은 계산량으로 축을 맞춘다.
- **기록하지 않은 것을 나중에 찾는 것.** 발산의 원인을 찾으려는데 gradient norm을 기록하지 않았다면 다시 학습해야 한다. 스칼라 몇 개는 처음부터 기록한다.

## 통과 기준

- 학습 스크립트가 손실·학습률·gradient norm·업데이트 비율·검증 metric을 적절한 주기로 기록하고, 원격 서버의 학습을 TensorBoard로 실시간 관찰한다.
- 학습 곡선(smoothing 없는 원본 포함)만 보고 과적합·과소적합·학습률 문제 중 무엇인지 근거와 함께 진단하고, 다음 실험의 조정 방향을 제안한다.
- 대시보드의 검증 metric과 독립 평가 스크립트의 결과를 대조해 일치를 확인한다.
- 연습문제 `debiased-ema`의 테스트를 통과한다.
