---
id: deep-learning-pytorch/evaluation-and-checkpoints
title: "평가·추론·체크포인트"
title_en: "Evaluation, Inference, and Checkpointing"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/training-loop]
---

# 평가·추론·체크포인트

학습 루프(17.5장)가 손실을 줄이는 과정이라면, 이 장은 그 결과를 **숫자 하나로 보고하고 파일 하나로 보존하는** 과정을 다룬다. 둘 다 학습보다 단순해 보여서 덜 점검된다. 그런데 논문의 표에 들어가는 것은 학습 손실이 아니라 평가 숫자이고, 며칠짜리 학습이 서버 점검으로 끊기면 그 뒤의 결과는 checkpoint가 얼마나 정확했는지에 달려 있다.

이 장의 세 주제에는 공통점이 있다. 모두 **틀려도 오류가 나지 않는다.** 평가 모드를 빠뜨려도, batch 평균을 다시 평균해도, optimizer 상태 없이 재개해도 코드는 끝까지 돌고 그럴듯한 숫자를 낸다. 그래서 이 장은 각 주제마다 "맞게 했는지 확인하는 방법"을 함께 다룬다. 6.6장에서 이런 버그를 잡는 테스트를 썼다면, 여기서는 그 테스트가 지키려던 성질이 PyTorch에서 무엇인지 본다.

## 학습 목표

- 평가 모드와 gradient 모드를 구분한다
- metric을 올바르게 집계한다
- 학습을 정확히 재개한다

## 평가 모드

평가 숫자는 같은 모델, 같은 데이터에 대해 언제 계산해도 같아야 한다. 학습할 때만 의미가 있는 무작위성이나 batch 의존성이 평가에 섞이면, 같은 checkpoint가 실행할 때마다 다른 점수를 받는다.

PyTorch에는 이와 관련된 스위치가 **두 개** 있고, 서로 독립이다.

- **모듈 모드**: `model.train()`과 `model.eval()`. 모델과 모든 하위 모듈의 `training` 플래그를 바꾼다. 이 플래그를 보고 동작을 바꾸는 층만 영향을 받는다. 대표적인 것이 dropout과 BatchNorm이다.
- **gradient 모드**: `torch.no_grad()`와 `torch.inference_mode()`. autograd가 연산을 그래프에 기록할지를 정한다. 층의 동작(출력값)은 바꾸지 않는다.

| | `train()` | `eval()` |
|---|---|---|
| **기록함** (기본) | 학습 step | 평가 결과는 맞지만 그래프를 만들어 메모리·시간을 낭비 |
| **기록 안 함** (`no_grad`) | dropout이 켜지고 BatchNorm이 평가 데이터로 running 통계를 갱신 | 평가의 올바른 조합 |

두 층의 동작을 식으로 쓰면 차이가 분명해진다. 확률 $p$의 dropout은 학습 모드에서 각 원소를 확률 $p$로 0으로 만들고 남은 원소를 $1/(1-p)$배 한다. 그래서 기댓값은 입력과 같고, 평가 모드에서는 아무것도 하지 않는 항등 함수다. BatchNorm은 학습 모드에서 **현재 batch**의 평균 $\mu_B$와 분산 $\sigma_B^2$로 정규화하고, 동시에 평가용 이동 평균을 갱신한다.

$$
\hat{\mu} \leftarrow (1 - m)\,\hat{\mu} + m\,\mu_B, \qquad \hat{\sigma}^2 \leftarrow (1 - m)\,\hat{\sigma}^2 + m\,\sigma_B^2 .
$$

여기서 $m$은 momentum(PyTorch 기본값 0.1)이고, $\hat{\mu}, \hat{\sigma}^2$는 parameter가 아니라 buffer로 저장되는 running 통계다(17.3장). 평가 모드에서는 $\mu_B, \sigma_B^2$ 대신 이 running 통계로 정규화한다. 이 갱신은 gradient와 상관없는 단순 대입이므로 **`no_grad` 안에서도 일어난다.**

```python {exec="python"}
import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(),
                      nn.Dropout(0.5), nn.Linear(8, 1))
x = torch.randn(16, 4)                                   # (B, D)
bn = model[1]

with torch.no_grad():                                    # gradient 기록만 끈다. 모드는 train 그대로
    y1, y2 = model(x), model(x)                          # (B, 1)
print("no_grad + train: 두 번 호출이 같은가 ", torch.equal(y1, y2))
print("no_grad + train: running_mean이 바뀌었는가", bool(bn.running_mean.abs().sum() > 0))

model.eval()                                             # dropout 끔, BN은 저장된 running 통계 사용
y3, y4 = model(x), model(x)
print("eval: 두 번 호출이 같은가          ", torch.equal(y3, y4))
print("eval: 출력이 그래프에 기록되는가   ", y3.requires_grad)
print("eval: 표본 0의 출력이 batch와 무관한가", torch.equal(model(x[:4])[0], y3[0]))

with torch.inference_mode():                             # 기록도 끄고 버전 추적도 생략
    y5 = model(x)
print("inference_mode: requires_grad", y5.requires_grad, " 값이 eval 출력과 같은가", torch.equal(y5, y3.detach()))
```

```output
no_grad + train: 두 번 호출이 같은가  False
no_grad + train: running_mean이 바뀌었는가 True
eval: 두 번 호출이 같은가           True
eval: 출력이 그래프에 기록되는가    True
eval: 표본 0의 출력이 batch와 무관한가 True
inference_mode: requires_grad False  값이 eval 출력과 같은가 True
```

결과를 한 줄씩 읽는다.

- `no_grad`만 쓰고 학습 모드로 두면 같은 입력의 두 출력이 다르다(dropout). running 통계도 평가 데이터로 바뀌었다. 평가가 모델을 **조용히 바꾼** 것이다.
- `eval()` 후에는 두 출력이 같고, 표본 0의 출력이 batch에 무엇이 함께 들어 있는지와 무관하다. 평가 숫자가 batch 크기에 의존하지 않는다는 뜻이다.
- 그러나 `eval()`만으로는 `requires_grad`가 여전히 `True`다. 모드 전환은 gradient 기록을 끄지 않는다.

::quiz{id="eval-does-not-disable-grad"}

**`no_grad`와 `inference_mode`.** 둘 다 기록을 끈다. `inference_mode`는 여기에 더해 텐서의 view 관계와 버전 카운터(in-place 수정을 감지하는 장치) 관리를 생략해 조금 더 가볍다. 대가로, `inference_mode` 안에서 만든 텐서는 나중에 autograd가 추적하는 계산에 넣을 수 없다. 그런 텐서를 역전파에 필요한 값으로 쓰면 오류가 난다. 평가·추론처럼 결과가 다시 학습으로 돌아가지 않는 코드에는 `inference_mode`를, 결과의 일부가 이후 학습 계산에 들어갈 수 있는 코드에는 `no_grad`를 쓴다.

연구 코드에서 지킬 규칙은 단순하다. **평가 함수가 스스로 `model.eval()`을 호출하고, 학습 루프는 매 학습 구간 시작에 `model.train()`을 호출한다.** 호출하는 쪽이 모드를 맞춰 줄 것이라고 가정하지 않는다. 평가 뒤에 학습으로 돌아갈 때 `train()`을 빠뜨리면 이번에는 학습에서 dropout이 꺼진다. 일부러 평가에서 dropout을 켜 두는 경우(Monte Carlo dropout으로 불확실성을 추정하는 방법, 21.1장)는 그렇게 한다는 사실을 코드와 기록에 명시한다.

## metric 집계

보고하는 metric은 평가 데이터 **전체**에 대한 숫자다. 그런데 계산은 메모리 때문에 batch로 나뉘어 일어난다. 나뉜 계산을 다시 합치는 방법이 틀리면, 같은 모델이 batch 크기에 따라 다른 점수를 받는다.

**정의.** 표본별 손실 $\ell_i$의 평균 metric은 $\frac{1}{N}\sum_{i=1}^{N}\ell_i$다. batch $B_k$의 크기를 $n_k$, batch 평균을 $\bar{\ell}_k$라 하면

$$
\frac{1}{N}\sum_{i=1}^{N}\ell_i = \sum_{k}\frac{n_k}{N}\,\bar{\ell}_k \quad\ne\quad \frac{1}{K}\sum_{k=1}^{K}\bar{\ell}_k \;\;(\text{일반적으로}).
$$

왼쪽은 batch 평균을 **크기로 가중**한 평균이고, 오른쪽의 "batch 평균의 평균"은 모든 $n_k$가 같을 때만 같다. 마지막 batch가 작으면 그 batch의 표본이 과대 대표된다. 가장 안전한 구현은 평균을 저장하지 않고 **합과 개수를 따로 누적**한 뒤 마지막에 한 번 나누는 것이다.

평균이 아닌 metric은 무엇을 누적할지가 metric마다 다르다.

| metric | 누적할 것 | 마지막 계산 |
|---|---|---|
| MAE, 정확도 | 오차(정답 수)의 합, 개수 | 합 / 개수 |
| RMSE | 제곱 오차의 합, 개수 | $\sqrt{\text{합}/\text{개수}}$ (batch RMSE를 평균하지 않는다) |
| $R^2$ | $\sum y$, $\sum y^2$, $\sum (y - \hat{y})^2$, 개수 | $1 - \text{SSE}/\text{SST}$ |
| precision, recall, F1 | 혼동행렬의 칸별 개수 | 마지막에 비율 계산 |
| AUC, 중앙값, 분위수 | 모든 예측과 정답 | 전체를 모아 한 번에 |

**mask.** 길이가 다른 시계열·문장·분자를 batch로 묶으면 짧은 것을 padding으로 채우고, 실제 관측 위치를 mask로 표시한다. 그러면 평균을 내는 단위를 정해야 한다. 관측값 하나하나를 같은 무게로 보는 **관측 단위 평균**과, 시계열마다 평균을 낸 뒤 시계열을 같은 무게로 보는 **시계열 단위 평균**은 서로 다른 질문에 대한 답이다. 둘 다 맞는 metric일 수 있다. 틀린 것은 padding을 관측처럼 세는 것뿐이다.

```python {exec="browser" session="metric"}
import numpy as np

rng = np.random.default_rng(0)
abs_err = np.sort(np.abs(rng.normal(size=1000)))         # (N,) 평가 데이터가 난이도 순으로 놓인 경우
batches = [abs_err[i:i + 64] for i in range(0, 1000, 64)]
print("batch 수", len(batches), " 마지막 batch 크기", len(batches[-1]))

total_sum, total_n = 0.0, 0
for b in batches:                                        # 합과 개수를 따로 누적한다
    total_sum, total_n = total_sum + b.sum(), total_n + b.size
print(f"batch 평균의 평균 {np.mean([b.mean() for b in batches]):.4f}")
print(f"합 / 개수         {total_sum / total_n:.4f}   (전체 한 번에 {abs_err.mean():.4f})")

# 길이가 다른 시계열 3개를 길이 5로 padding했다. mask = 1이 실제 관측
err = np.array([[1., 1., 1., 1., 1.],
                [2., 2., 0., 0., 0.],
                [4., 0., 0., 0., 0.]])                   # (B, T)
mask = (np.arange(5) < np.array([[5], [2], [1]])).astype(float)   # (B, T)
print(f"padding 포함 평균       {err.mean():.4f}")
print(f"관측 단위 평균          {(err * mask).sum() / mask.sum():.4f}")
print(f"시계열 단위 평균의 평균 {((err * mask).sum(1) / mask.sum(1)).mean():.4f}")
```

```output
batch 수 16  마지막 batch 크기 40
batch 평균의 평균 0.8166
합 / 개수         0.7772   (전체 한 번에 0.7772)
padding 포함 평균       0.8667
관측 단위 평균          1.6250
시계열 단위 평균의 평균 2.3333
```

평가 데이터가 난이도 순으로 놓여 있어 마지막 batch(40개)에 가장 어려운 표본이 모였다. batch 평균의 평균은 이 40개를 64개처럼 셈해 0.8166을 냈고, 올바른 값 0.7772보다 약 5% 크다. 셔플한 평가 데이터에서는 차이가 작아지지만, 평가 loader는 보통 셔플하지 않고 데이터 파일은 흔히 시간·시료·난이도 순으로 저장되어 있다. 아래 세 줄은 padding을 0으로 세면 평균이 절반 가까이 줄고, 관측 단위(1.625)와 시계열 단위(2.333)가 크게 다르다는 것을 보여 준다. 짧은 시계열의 오차가 클 때 두 정의의 차이가 커진다.

::quiz{id="mean-of-batch-means"}

:::note[분야 예시]
기후 모델의 격자 출력을 평가할 때 위도-경도 격자의 칸은 넓이가 같지 않다. 고위도로 갈수록 칸의 실제 넓이가 줄어들므로, 전 지구 평균 오차는 보통 각 칸에 위도의 cosine에 비례하는 가중치를 주어 계산한다. 여기에 육지·해양 mask가 겹치면 "가중치 × mask의 합"이 분모가 된다. 가중치와 mask를 batch마다 따로 정규화하면 위의 batch 평균의 평균과 같은 문제가 생긴다. 분자와 분모를 끝까지 따로 누적한다.
:::

여러 GPU로 평가를 나누어 실행할 때도 원리는 같다. 각 프로세스의 합과 개수를 모아(all-reduce) 마지막에 나눈다. 데이터를 프로세스 수로 나누어떨어지게 하려고 표본을 일부 중복시키는 sampler가 있으므로, 중복 표본이 metric에 들어가는지 확인한다(22.6장).

## 체크포인트의 내용

checkpoint(체크포인트)는 학습의 한 시점을 파일로 남긴 것이다. 무엇을 남겨야 하는지는 그 파일로 무엇을 할지에 달려 있다. 평가만 하려면 모델 가중치로 충분하지만, **학습을 정확히 이어 가려면** 다음 step의 결과에 영향을 주는 모든 상태가 필요하다.

| 항목 | 내용 | 빠지면 |
|---|---|---|
| 모델 `state_dict` | parameter와 buffer(BatchNorm running 통계 등) | 모델 자체가 없다 |
| optimizer `state_dict` | Adam의 1·2차 moment, step 수, 현재 학습률 등 param group 설정 | 재개 직후 momentum이 0에서 다시 시작한다 |
| scheduler `state_dict` | 현재 step, 기준 학습률 | 학습률이 처음 값으로 돌아간다 |
| `GradScaler` 상태 | mixed precision의 loss scale | scale을 다시 찾는 동안 step이 건너뛰어진다 |
| RNG 상태 | PyTorch CPU·CUDA, NumPy, Python `random`, 데이터 표집용 generator | dropout·셔플·증강의 난수열이 달라진다 |
| 진행 정보 | step, epoch, 지금까지의 best metric | 스케줄·early stopping·로그가 어긋난다 |
| 설정과 버전 | 해석이 끝난 설정(6.3장), 코드 commit, 데이터 버전 | 무엇을 학습한 파일인지 알 수 없다 |

Adam 계열 optimizer는 parameter마다 moment 두 개를 저장하므로, 전체 checkpoint는 가중치만 저장한 파일의 약 세 배가 된다. 그래서 재개용 checkpoint와 배포·평가용 가중치 파일을 따로 두는 경우가 많다.

PyTorch에서 모델·optimizer·scheduler는 모두 `state_dict()`와 `load_state_dict()`를 제공한다. 작은 학습으로 각 항목이 실제로 결과에 관여하는지 확인한다. 먼저 끊김 없이 40 step 학습한 결과를 기준으로 남긴다.

```python {exec="python"}
import io

def make(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    return model, opt, sched

def train(model, opt, sched, steps):
    model.train()
    for _ in range(steps):
        x = torch.randn(32, 3)                           # (B, 3) 데이터 표집도 전역 RNG를 쓴다
        y = x.sum(1, keepdim=True)                       # (B, 1)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

m, o, s = make(seed=0)
train(m, o, s, 40)
straight = flat(m)                                       # 끊김 없이 40 step
print("파라미터 수", straight.numel(), " 마지막 학습률", s.get_last_lr()[0])
```

```output
파라미터 수 81  마지막 학습률 0.0
```

이제 20 step에서 저장하고, 다른 시드로 만든 새 객체에 불러와 나머지 20 step을 학습한다. 새 객체는 "다음 날 새 프로세스에서 다시 시작한 학습"을 흉내 낸다. 복원할 항목을 하나씩 빼 본다.

```python {exec="python"}
def resume(restore):
    m, o, s = make(seed=0)
    train(m, o, s, 20)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "optimizer": o.state_dict(), "scheduler": s.state_dict(),
                "rng": torch.get_rng_state(), "step": 20}, buf)
    buf.seek(0)
    ckpt = torch.load(buf)
    m, o, s = make(seed=123)                             # 새 프로세스라고 생각한다
    m.load_state_dict(ckpt["model"])
    if "optimizer" in restore: o.load_state_dict(ckpt["optimizer"])
    if "scheduler" in restore: s.load_state_dict(ckpt["scheduler"])
    if "rng" in restore: torch.set_rng_state(ckpt["rng"])
    train(m, o, s, 40 - ckpt["step"])
    return flat(m)

full = {"optimizer", "scheduler", "rng"}
for name, restore in [("전부 복원", full), ("optimizer 빠짐", full - {"optimizer"}),
                      ("scheduler 빠짐", full - {"scheduler"}), ("RNG 빠짐", full - {"rng"})]:
    p = resume(restore)
    print(f"{name:14s} 일치 {torch.equal(p, straight)!s:5s}  최대 차이 {(p - straight).abs().max():.2e}")
```

```output
전부 복원          일치 True   최대 차이 0.00e+00
optimizer 빠짐   일치 False  최대 차이 7.38e-02
scheduler 빠짐   일치 False  최대 차이 4.58e-02
RNG 빠짐         일치 False  최대 차이 1.01e-02
```

전부 복원했을 때만 최종 파라미터가 **비트 단위로** 같다. optimizer 상태가 빠지면 Adam의 moment가 0에서 다시 쌓이고, scheduler가 빠지면 학습률이 처음 값으로 돌아가고, RNG가 빠지면 dropout mask와 데이터 표집이 달라진다. 셋 다 오류 없이 돌고 손실 곡선도 그럴듯하게 이어진다는 점이 중요하다. 곡선을 보고는 알 수 없고, 비교 테스트로만 드러난다(6.6장).

저장과 불러오기에서 지킬 점이 몇 가지 있다.

- **`torch.save`는 Python pickle 형식이다.** pickle은 불러올 때 임의의 코드를 실행할 수 있으므로 출처를 모르는 파일을 열지 않는다. PyTorch 2.6부터 `torch.load`의 기본값은 텐서와 기본 자료형만 읽는 `weights_only=True`다(집필 시점 예시. PyTorch 2.15에서 확인). NumPy RNG 상태처럼 다른 객체가 든 checkpoint는 이 모드로 열리지 않으므로, 직접 만든 파일에 한해 `weights_only=False`를 명시한다.
- **저장은 원자적으로 한다.** 저장 도중 프로세스가 죽으면 반쯤 쓰인 파일이 이전의 멀쩡한 `last.pt`를 덮어쓴다. 임시 파일에 쓴 뒤 `os.replace`로 이름을 바꾸면 이 위험이 없다.
- **GPU에서 저장한 파일을 CPU에서 열 때는** `map_location="cpu"`를 준다.
- **DataLoader의 epoch 중간 위치는 기본으로 저장되지 않는다.** epoch 경계에서만 저장하거나, 셔플 순서를 시드와 epoch 번호로 결정해 재개할 때 이미 본 batch를 건너뛰거나, 실습처럼 표집용 generator의 상태를 저장한다.

## best와 last

학습 중 저장하는 checkpoint는 용도가 둘이다.

- **last**: 가장 최근 상태. 재개용이다. 일정 간격으로 덮어쓴다.
- **best**: 미리 정한 **검증** metric이 가장 좋았던 상태. 보고·배포 후보다.

best를 고르는 기준은 학습을 시작하기 전에 정한다. metric 하나(예: 검증 MAE), 방향(작을수록 좋음), 개선으로 인정할 최소 폭 `min_delta`다. **test 데이터는 고르는 데 쓰지 않는다.** 고르는 데 쓴 데이터의 숫자는 낙관적으로 치우치기 때문이다(15.4장). 이 낙관이 검증 데이터에서도 생긴다는 것을 작은 모의 실험으로 본다. 기저 곡선은 epoch 22에서 가장 낮지만 관측한 검증 손실에는 잡음이 있다. 검증 손실이 5 epoch 동안 `min_delta` 이상 좋아지지 않으면 멈추는 early stopping을 쓴다.

```python {exec="browser" session="metric"}
rng = np.random.default_rng(1)
epochs = np.arange(1, 61)
true_val = 0.5 + 0.5 * np.exp(-epochs / 8) + 0.004 * epochs   # (60,) 내려갔다 다시 오르는 기저 곡선
val = true_val + rng.normal(0, 0.02, size=60)                  # 관측한 검증 손실
test = true_val + rng.normal(0, 0.02, size=60)                 # 같은 epoch의 독립된 test 손실

best, best_epoch, wait, patience, min_delta = np.inf, 0, 0, 5, 1e-3
for e, v in zip(epochs, val):
    if v < best - min_delta:                             # 의미 있게 좋아졌을 때만 best 갱신
        best, best_epoch, wait = v, e, 0
    else:
        wait += 1
        if wait >= patience:
            break
print(f"정지 epoch {e}, best epoch {best_epoch}, best의 검증 손실 {best:.3f}")
print(f"best epoch의 test 손실 {test[best_epoch - 1]:.3f}, 기저 곡선 {true_val[best_epoch - 1]:.3f}")
print(f"기저 곡선이 가장 낮은 epoch {epochs[true_val.argmin()]} (값 {true_val.min():.3f})")
```

```output
정지 epoch 24, best epoch 19, best의 검증 손실 0.607
best epoch의 test 손실 0.620, 기저 곡선 0.623
기저 곡선이 가장 낮은 epoch 22 (값 0.620)
```

멈춘 시점(epoch 24)과 best(epoch 19)가 다르다. 그래서 early stopping은 last가 아니라 best를 복원해야 한다. 고른 epoch의 검증 손실(0.607)은 그 epoch의 기저 곡선(0.623)보다 낮다. 여러 epoch 중 **잡음이 유리하게 나온 것**을 골랐기 때문이다. 고르는 데 쓰지 않은 test 손실(0.620)은 기저 곡선에 가깝다. 선택에 쓴 숫자를 그대로 최종 성능으로 보고하면 안 되는 이유다. 비교할 후보(epoch, 하이퍼파라미터)가 많을수록 이 치우침은 커진다.

::quiz{id="best-checkpoint-selection"}

저장 정책은 디스크와 잃을 수 있는 시간 사이의 거래다.

- last는 자주 덮어쓴다. 저장 간격이 $\Delta$ step이면 중단될 때 평균 $\Delta/2$ step을 다시 학습한다. 공유 클러스터처럼 작업이 시간 제한으로 끊기는 환경(2.5장)에서는 제한 시간 전에 저장하도록 만든다.
- best는 검증할 때마다 비교해 갱신한다. 검증 간격보다 짧게 저장할 필요는 없다.
- 학습 동역학을 나중에 분석하려면 일정 간격의 snapshot을 따로 남기되, 개수를 제한한다.
- 파일 이름이나 함께 저장한 메타데이터에 step과 metric을 남긴다. `best.pt`만으로는 어느 시점인지 알 수 없다.

## 재개와 추론

같은 checkpoint를 여는 목적이 둘이고, 필요한 것이 다르다.

**학습 재개.** 위 표의 모든 상태를 복원하고, 같은 설정·같은 코드로 남은 step을 진행한다. 재개할 때 총 step 수나 batch 크기를 바꾸면 scheduler의 스케줄과 step의 의미가 달라지므로 더 이상 "같은 학습의 연속"이 아니다. 그런 변경은 새 실험으로 기록한다. 재개가 정확한지는 끊김 없는 짧은 학습과 비교해 확인한다(실습).

**추론과 평가.** 모델 구조를 설정으로 다시 만들고, 가중치만 불러온 뒤, `eval()`과 `inference_mode`로 실행한다. optimizer·scheduler·RNG는 필요 없다. 학습 코드를 import하지 않고 이 일을 할 수 있어야 한다(6.1장의 `eval.py`, 17.7장).

**가중치에서 시작하는 새 학습.** 사전학습한 가중치로 다른 과제를 학습하는 fine-tuning은 그 중간이다. 가중치는 불러오지만 optimizer와 scheduler는 새로 만든다. 출력층이 바뀌어 이름이나 shape이 맞지 않으면 `load_state_dict(..., strict=False)`를 쓰는데, 이때 반환값의 `missing_keys`와 `unexpected_keys`를 반드시 출력해 확인한다. 이름이 조금 달라(예: 분산 학습 wrapper가 붙인 접두사) 아무 가중치도 불러오지 못했는데 `strict=False` 때문에 오류 없이 무작위 초기화 상태로 학습하는 일이 흔하다.

## inference contract

모델 가중치만으로는 예측을 재현할 수 없다. 입력이 어떤 순서·단위로 들어가고, 어떤 통계로 정규화되고, 출력을 어떻게 원래 단위로 되돌리는지가 모두 예측의 일부다. 이것을 문서화한 명세를 이 책에서는 **inference contract**라 부른다. contract는 checkpoint 안에 함께 저장하는 것이 가장 안전하다. 실습의 checkpoint에는 다음이 들어 있다.

```json
{
  "inputs": ["temp_k", "pressure_bar"],
  "x_mean": [322.53, 2.97], "x_std": [23.18, 1.13],
  "y_mean": 1.65, "y_std": 1.14
}
```

(숫자는 표시를 위해 반올림했다.) 빠지기 쉬운 항목은 다음과 같다.

- **정규화 통계.** 학습 데이터에서 계산한 평균·표준편차다. 추론할 때 새 입력으로 다시 계산하면 안 된다. 입력이 하나뿐이면 표준편차가 0이 되고, 입력이 여러 개여도 분포가 학습 때와 다르면 같은 원시값이 다른 표준화 값이 된다. 평가 데이터로 통계를 계산하면 누수이기도 하다(15.4장).
- **입력의 순서와 단위.** 켈빈과 섭씨, bar와 Pa는 코드 오류 없이 예측을 망친다.
- **출력의 역변환.** 표준화한 목표값, 로그를 취한 목표값을 원래 단위로 되돌리는 방법.
- **결측과 범위 밖 입력의 처리.** 학습 데이터 범위(예: 280–360 K)를 함께 저장하면, 범위 밖 입력에 경고를 낼 수 있다.
- **모델 설정과 코드 버전.** 같은 이름의 모델 클래스가 코드 변경으로 다른 구조가 되는 경우를 막는다.

:::note[분야 예시]
분자 물성 예측 모델은 보통 SMILES 문자열을 정규화(canonicalization)하고, 수소 처리·전하 상태를 정한 뒤 특징을 계산한다. 같은 분자라도 이 전처리 규칙이 다르면 다른 입력이 된다. 실험실 장비의 스펙트럼 모델은 기준선 보정과 파장 격자 보간을 거친다. 이런 규칙은 모델 파일 밖의 전처리 코드에 있으므로, contract에 전처리 함수의 이름과 버전을 남기지 않으면 몇 달 뒤 같은 예측을 재현할 수 없다.
:::

contract를 실제 서비스 형태로 내보내는 일(입력 검증, 배포 형식)은 22.7장에서 다룬다.

## 실습

학습 중간에 중단·재개한 결과가 끊김 없이 학습한 결과와 일치하는지 검증하고, 독립 평가 스크립트로 결과를 재현한다. 코드는 `labs/deep-learning-pytorch/evaluation-and-checkpoints/`에 있다.

- `common.py`: 합성 회귀 데이터(온도 K, 압력 bar → 물성값), 모델 정의, 평가 함수(`eval()` + `inference_mode` + 합/개수 누적. 검증 데이터 256개를 100개씩 나누므로 마지막 batch는 56개다).
- `train.py`: AdamW + cosine scheduler + dropout. 50 step마다 검증해 `best.pt`를 갱신하고, 끝나면 `last.pt`를 저장한다. checkpoint에는 모델·optimizer·scheduler·RNG 네 종류·진행 정보·설정·contract가 들어간다. `--stop-at`으로 중단을, `--resume`으로 재개를 흉내 낸다.
- `eval.py`: checkpoint 경로만 받는다. 학습 루프와 optimizer를 import하지 않고, 설정으로 모델을 다시 만들고 contract로 전처리한다. 파라미터의 해시를 함께 출력한다.

```bash
cd labs/deep-learning-pytorch/evaluation-and-checkpoints
uv run --project ../.. --group torch python train.py --out outputs/straight
uv run --project ../.. --group torch python train.py --out outputs/resumed --stop-at 120
uv run --project ../.. --group torch python train.py --out outputs/resumed --resume outputs/resumed/last.pt
uv run --project ../.. --group torch python eval.py outputs/straight/last.pt outputs/resumed/last.pt outputs/straight/best.pt
```

집필 시점의 실행 결과(CPU, PyTorch 2.15)는 다음과 같다.

```output
$ python train.py --out outputs/straight
step   50  train loss 0.0898  검증 MAE 0.3175
step  100  train loss 0.0377  검증 MAE 0.2185
step  150  train loss 0.0681  검증 MAE 0.1776
step  200  train loss 0.0441  검증 MAE 0.1468
step  250  train loss 0.0255  검증 MAE 0.1498
step  300  train loss 0.0216  검증 MAE 0.1492
저장: straight/last.pt (step 300)
$ python train.py --out outputs/resumed --stop-at 120
step   50  train loss 0.0898  검증 MAE 0.3175
step  100  train loss 0.0377  검증 MAE 0.2185
저장: resumed/last.pt (step 120)
$ python train.py --out outputs/resumed --resume outputs/resumed/last.pt
재개: step 120, best 검증 MAE 0.2185
step  150  train loss 0.0681  검증 MAE 0.1776
step  200  train loss 0.0441  검증 MAE 0.1468
step  250  train loss 0.0255  검증 MAE 0.1498
step  300  train loss 0.0216  검증 MAE 0.1492
저장: resumed/last.pt (step 300)
$ python eval.py outputs/straight/last.pt outputs/resumed/last.pt outputs/straight/best.pt
straight/last.pt     step  300  검증 MAE 0.1492  파라미터 해시 2d6991b93680
resumed/last.pt      step  300  검증 MAE 0.1492  파라미터 해시 2d6991b93680
straight/best.pt     step  200  검증 MAE 0.1468  파라미터 해시 a57f306025bd
```

120 step에서 끊고 재개한 학습은 150 step 이후의 모든 숫자가 끊김 없는 학습과 같고, 최종 파라미터 해시도 같다. 독립된 `eval.py`가 학습 스크립트의 마지막 검증 MAE(0.1492)를 그대로 재현한다. best는 step 200에서 저장되었고 그 뒤 검증 MAE가 조금 올랐다. 이 결과를 자신의 코드로 확인한 뒤 다음을 시도한다.

1. `train.py`의 `save()`에서 `"optimizer"`를 빼고(불러오는 쪽도) 다시 실행해 해시가 달라지는 것을 확인한다.
2. `common.py`의 `evaluate`에서 `model.eval()`을 지우고 `eval.py`를 두 번 실행한다. 같은 checkpoint가 실행마다 다른 MAE를 내는지 본다.
3. GPU에서 실행한다면 비트 단위 일치가 깨질 수 있다. 비결정적 커널 때문이며, 22.2장에서 결정적 실행을 다룬다. 이때는 허용 오차 안의 일치를 기준으로 삼는다.

이어서 두 연습문제를 푼다.

::exercise{id="masked-mae-accumulator"}

::exercise{id="resumable-sgd"}

## 실패 사례

- **평가에서 `model.eval()`을 빠뜨리는 것.** dropout이 켜진 채로 평가하면 점수가 실행마다 흔들리고 보통 나빠진다. BatchNorm은 평가 batch의 통계로 정규화하므로 점수가 평가 batch 크기에 따라 달라지고, running 통계가 평가 데이터로 오염되어 **이후 학습과 평가까지** 영향을 받는다. `no_grad`는 이것을 막지 못한다.
- **마지막 batch의 크기를 무시한 평균.** batch 평균을 다시 평균하거나, 평가 loader에 `drop_last=True`를 두어 마지막 표본을 버리는 것. 두 경우 모두 오류 없이 batch 크기에 따라 다른 숫자가 나온다. 평가 함수는 batch 크기 1, 7, 전체 크기에서 같은 값을 내는지 테스트한다.
- **가중치만 저장하고 재개하는 것.** 재개 직후 손실이 잠깐 튀거나 아무 일 없어 보이지만, 결과는 끊김 없는 학습과 다르다. 논문에 "중간에 재개함"이 들어간 실험과 그렇지 않은 실험을 비교하면, 차이의 일부는 재개 방식에서 온다.
- **best를 test 점수로 고르거나, best의 검증 점수를 최종 성능으로 보고하는 것.** 고르는 데 쓴 숫자는 낙관적이다.
- **정규화 통계를 저장하지 않는 것.** 추론 코드가 새 데이터로 통계를 다시 계산하거나, 통계 없이 원시값을 넣는다. 모델은 아무 경고 없이 다른 함수가 된다.

## 통과 기준

- 학습을 중간에 중단·재개한 결과가 끊김 없는 학습과 같음을 파라미터 비교(CPU에서는 비트 단위)로 보인다.
- checkpoint 복원 후 학습 코드를 import하지 않는 독립 평가 스크립트로 학습 중 기록한 metric과 같은 숫자를 재현한다.
- 평가 함수가 batch 크기와 무관하게 같은 metric을 낸다는 테스트를 쓴다.
- 연습문제 두 개(`masked-mae-accumulator`, `resumable-sgd`)의 테스트를 통과한다.
