---
id: deep-learning-pytorch/trainer-design
title: "Trainer 설계: 직접 작성과 프레임워크"
title_en: "Designing Trainers: Hand-Written vs Frameworks"
tier: core
status: draft
prerequisites: [deep-learning-pytorch/evaluation-and-checkpoints, research-code-engineering/configuration]
---

# Trainer 설계: 직접 작성과 프레임워크

17.5장의 학습 루프는 한 파일 안에서 끝났다. 실험이 늘어나면 같은 루프를 복사해 조금씩 고친 파일이 쌓인다. 한 파일에서 고친 버그가 다른 파일에는 남고, 비교하려던 두 실험이 사실은 루프의 세부(검증 주기, clipping, 저장 기준)에서도 달라진다. 반복되는 부분을 **trainer**로 모으는 이유다.

반대 방향의 위험도 있다. 모든 경우를 처리하려는 trainer는 옵션이 수십 개로 늘어 아무도 전체를 이해하지 못하게 되고, 남이 만든 프레임워크는 이해하지 못한 기본값으로 결과를 바꾼다. 이 장은 직접 쓰는 trainer가 맡을 책임의 경계를 정하고, 그 경계를 기준으로 프레임워크를 언제 도입할지 판단하는 방법을 다룬다. 설정을 코드에서 분리하는 방법(6.3장)과 checkpoint의 내용(17.6장)은 이미 안다고 가정한다.

## 학습 목표

- 재사용 가능한 trainer.py를 설계한다
- 학습 프레임워크를 언제 도입할지 판단한다

## trainer.py의 책임

trainer가 무엇을 하는지보다 **무엇을 하지 않는지**를 먼저 정하는 것이 설계의 핵심이다. 6.1장에서 책임의 경계를 "데이터가 넘어가는 지점"으로 정했던 것처럼, trainer와 나머지 코드의 경계도 주고받는 것으로 정의한다.

| 맡는다 (모든 실험에서 같다) | 맡지 않는다 (실험마다 다르다) |
|---|---|
| step의 순서: `zero_grad → backward → (clip) → step → scheduler` | 모델 구조 |
| epoch·step 세기, 정해진 주기의 검증 호출 | 데이터와 batch 구성 |
| callback 호출 | loss의 정의, 과제별 forward |
| 학습 상태의 저장·복원 | metric의 정의와 집계 |
| 멈춤 신호 처리 | 설정 해석(6.3장) |

왼쪽은 실험이 바뀌어도 바뀌지 않아야 하는 부분이다. 여기를 한곳에 모으면 "두 실험이 루프의 세부에서 달랐다"는 가능성이 사라진다. 오른쪽은 연구 질문 그 자체이므로 trainer 밖에 둔다. 경계를 넘는 것은 함수 두 개다.

- **step 함수** `step_fn(model, batch) -> loss`: 과제별 forward와 loss.
- **평가 함수** `eval_fn(model) -> dict`: 평가 모드 전환, 합/개수 누적, metric 계산(17.6장).

이 경계를 NumPy로 먼저 그려 본다. 아래 `Trainer`는 반복문의 순서와 이벤트 발송만 알고, 선형 회귀든 무엇이든 손실과 업데이트는 `step_fn`이 정한다.

```python {exec="browser" session="trainer"}
import numpy as np

class Trainer:
    def __init__(self, step_fn, eval_fn, callbacks=(), max_epochs=100):
        self.step_fn, self.eval_fn = step_fn, eval_fn          # 과제별 코드는 밖에서 받는다
        self.callbacks, self.max_epochs = list(callbacks), max_epochs
        self.should_stop, self.epoch = False, 0

    def _emit(self, event, **info):
        for cb in self.callbacks:
            getattr(cb, event, lambda trainer, **kw: None)(self, **info)

    def fit(self, params, batches):
        for self.epoch in range(1, self.max_epochs + 1):
            for batch in batches:
                params, loss = self.step_fn(params, batch)     # forward·backward·update는 과제가 정한다
            self._emit("on_epoch_end", train_loss=loss)
            self._emit("on_eval_end", metric=self.eval_fn(params))
            if self.should_stop:
                break
        return params
```

```output

```

부가 기능은 **callback**으로 붙인다. callback은 trainer가 정해진 시점(이벤트)에 호출하는 객체다. trainer는 callback이 무엇을 하는지 모르고, callback은 trainer의 공개된 상태(`epoch`, `should_stop`)만 읽고 쓴다. 아래는 early stopping과 기록을 callback으로 만들고, 같은 trainer로 손실이 다른 두 과제(MSE, Huber)를 학습한다. 잡음이 두꺼운 꼬리(자유도 2의 t 분포)를 가져서 이상치에 덜 민감한 Huber 손실이 의미가 있는 상황이다.

```python {exec="browser" session="trainer"}
class EarlyStopping:
    def __init__(self, patience, min_delta=1e-3):
        self.patience, self.min_delta, self.best, self.wait = patience, min_delta, np.inf, 0
    def on_eval_end(self, trainer, metric):
        improved = metric < self.best - self.min_delta
        self.best, self.wait = (metric, 0) if improved else (self.best, self.wait + 1)
        trainer.should_stop = self.wait >= self.patience

class History:
    def __init__(self):
        self.rows = []
    def on_eval_end(self, trainer, metric):
        self.rows.append((trainer.epoch, round(float(metric), 4)))

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))                                  # (N, D)
y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_t(df=2, size=200) * 0.3   # 두꺼운 꼬리 잡음
batches = [(X[i:i + 20], y[i:i + 20]) for i in range(0, 160, 20)]
Xv, yv = X[160:], y[160:]

def mse_step(w, b, lr=0.05):
    r = b[0] @ w - b[1]
    return w - lr * 2 * b[0].T @ r / len(r), np.mean(r ** 2)
def huber_step(w, b, lr=0.05, delta=1.0):
    r = b[0] @ w - b[1]
    return w - lr * b[0].T @ np.clip(r, -delta, delta) / len(r), np.mean(np.minimum(r ** 2 / 2, delta * (np.abs(r) - delta / 2)))
val_mae = lambda w: np.mean(np.abs(Xv @ w - yv))

for name, step in [("MSE", mse_step), ("Huber", huber_step)]:
    hist = History()
    w = Trainer(step, val_mae, [EarlyStopping(patience=3), hist], max_epochs=50).fit(np.zeros(3), batches)
    print(f"{name:5s} 멈춘 epoch {hist.rows[-1][0]:2d}  검증 MAE {hist.rows[-1][1]:.4f}  w = {np.round(w, 3)}")
```

```output
MSE   멈춘 epoch 11  검증 MAE 0.3788  w = [ 1.08  -1.945  0.478]
Huber 멈춘 epoch 22  검증 MAE 0.3747  w = [ 0.972 -1.966  0.498]
```

`Trainer` 클래스는 한 줄도 바뀌지 않았다. 두 과제의 차이는 step 함수에만 있고, 멈춘 시점은 각 과제의 검증 곡선이 정했다. 이것이 재사용 가능한 trainer의 기준이다. **새 과제를 추가할 때 trainer 파일을 열 필요가 없어야 한다.**

PyTorch에서도 구조는 같다. 다만 이제 backward와 `optimizer.step()`의 순서가 trainer의 책임으로 들어온다. 이 순서는 17.5장에서 본 대로 틀리기 쉽고, 모든 과제에서 같기 때문이다. trainer로 옮긴 코드는 옮기기 전과 **같은 결과**를 내야 한다. 리팩터링이 결과를 바꾸지 않았다는 확인은 파라미터 비교로 한다.

```python {exec="python"}
import torch
from torch import nn

def fit(model, opt, loader, step_fn, max_epochs):
    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            loss = step_fn(model, batch)                       # 과제별: forward와 loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def regression_step(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)

def plain_loop(model, opt, loader, max_epochs):                # 17.5장의 루프를 그대로 쓴 것
    for epoch in range(max_epochs):
        model.train()
        for x, y in loader:
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def setup(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(64, 3)                                     # (N, 3)
    y = torch.sin(x[:, :1]) + x[:, 1:2] * x[:, 2:]              # (N, 1)
    data = list(zip(x.split(16), y.split(16)))                 # batch 4개
    model = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
    return model, torch.optim.Adam(model.parameters(), lr=1e-2), data

a = fit(*setup(), regression_step, max_epochs=200)
b = plain_loop(*setup(), max_epochs=200)
print("trainer와 직접 쓴 루프의 결과가 같은가:",
      all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))
```

```output
trainer와 직접 쓴 루프의 결과가 같은가: True
```

비트 단위로 같다. 이 비교는 trainer를 고칠 때마다 다시 돌리는 회귀 테스트(6.6장)로 남겨 둔다.

## 확장 지점

trainer를 재사용하다 보면 "이 실험에서만 필요한 기능"이 계속 생긴다. 매번 trainer에 `if` 플래그를 추가하면 trainer는 곧 모든 실험의 예외를 담은 거대한 함수가 된다. 확장 방법을 미리 정해 두면 이를 막을 수 있다. 확장 지점은 세 종류다.

- **step 함수 교체**: 과제마다 forward와 loss가 다를 때. 분류·회귀·자기지도 학습·물리 손실 항 추가(24.2장)가 여기에 속한다.
- **callback**: 루프의 흐름은 그대로 두고 특정 시점에 무언가를 **관찰하거나 저장하거나 멈추게** 할 때. 로깅, checkpoint, early stopping, 학습률 기록, 예측 샘플 저장, 일정 epoch 후 일부 층 동결 해제.
- **trainer 상속이나 새 trainer**: 루프의 **구조 자체**가 다를 때. 한 step에 optimizer가 둘인 학습(GAN), 안쪽·바깥쪽 루프가 있는 이중수준 최적화(10.7장), 환경과 상호작용하는 강화학습(20장)은 callback으로 흉내 내지 말고 별도 루프를 쓴다.

callback의 이벤트는 많을 필요가 없다. 실습의 `trainer.py`는 `on_train_start`, `on_step_end`, `on_eval_end`, `on_train_end` 네 개로 로깅·저장·조기 종료를 모두 처리한다. 이벤트를 추가하기 전에 기존 이벤트로 할 수 없는지 먼저 묻는다.

callback 설계에서 지킬 규칙이 둘 있다.

- **callback끼리 서로를 몰라야 한다.** checkpoint callback이 early stopping callback의 내부 값을 읽기 시작하면 순서 의존성이 생긴다. 공유가 필요한 값은 trainer의 상태(예: `last_metrics`)를 통해 주고받는다.
- **상태가 있는 callback은 `state_dict`를 가진다.** early stopping의 `best`와 기다린 횟수는 학습 상태의 일부다. 재개할 때 이것을 복원하지 않으면 재개 직후의 첫 검증이 무조건 "개선"으로 판정되고 patience가 처음부터 다시 센다. 실습의 trainer는 callback의 상태를 checkpoint에 함께 저장한다.

## eval.py의 분리

평가 코드는 학습 코드 없이 실행할 수 있어야 한다. 이유는 연구의 시간 순서에 있다.

- **재평가.** metric 계산의 버그를 고치거나 새 metric을 추가하면, 이미 학습한 checkpoint를 **다시 학습하지 않고** 모두 다시 평가해야 한다.
- **공정한 비교.** baseline, ablation, 다른 시드의 checkpoint를 같은 평가 코드로 평가해야 표의 숫자들이 비교 가능하다(6.1장).
- **새 데이터.** 몇 달 뒤 새로 측정한 데이터에 같은 모델을 적용할 때, 그때의 학습 코드는 이미 바뀌어 있을 수 있다.

그래서 `eval.py`가 import하는 것은 모델 정의, 데이터 읽기, metric, checkpoint 읽기뿐이어야 한다. trainer, optimizer, 학습 설정의 기본값에 의존하지 않는다. 학습 중에 계산한 검증 metric과 `eval.py`가 다시 계산한 값이 같은지 확인하는 것이 이 분리가 제대로 되었는지에 대한 시험이다. 실습의 `eval.py`는 일부러 학습 때와 다른 batch 크기(7)로 평가해, 집계가 batch 크기에 의존하지 않는지도 함께 확인한다.

::quiz{id="eval-script-separation"}

## 프레임워크

직접 쓴 trainer가 맡는 일을 대신 해 주는 학습 프레임워크가 여럿 있다. 차이는 **얼마나 많은 것을 가져가는가**, 즉 추상화 수준이다. 아래 설명은 각 도구의 역할을 일반적으로 정리한 것이다(집필 시점 예시. 이 책의 실습 환경에는 설치되어 있지 않으며, 구체적인 API와 기본값은 각 도구의 공식 문서에서 확인한다).

| 도구 | 사용자가 쓰는 것 | 도구가 가져가는 것 | 추상화 |
|---|---|---|---|
| 직접 쓴 trainer | 모든 것 | 없음 | 가장 낮음 |
| Hugging Face Accelerate | 학습 루프 전체 | 장치 배치, 분산 실행, mixed precision의 반복 코드 | 낮음 |
| PyTorch Lightning | 모델과 step 단위 메서드(학습·검증 step, optimizer 구성) | 루프, 장치·분산·정밀도, checkpoint, 로깅, callback 체계 | 높음 |
| Hugging Face Transformers의 `Trainer` | 모델, 데이터, 인자 묶음 | 루프 전체와 그 모델 계열에 맞춘 기본 레시피 | 가장 높음 |

- **Accelerate**는 루프를 사용자가 그대로 쓰게 두고, 같은 코드를 CPU·GPU 하나·여러 GPU에서 돌게 만드는 얇은 층이다. 직접 쓴 trainer와 함께 쓸 수 있다.
- **Lightning**은 이 장의 구조(trainer가 루프를 갖고, 사용자는 step을 정의하고, 부가 기능은 callback)와 같은 생각을 더 넓게 구현한 프레임워크다. 그래서 직접 trainer를 설계해 본 사람은 Lightning의 구조를 빨리 읽는다.
- **Transformers의 `Trainer`**는 그 라이브러리의 모델을 학습하는 데 맞춰져 있다. 그 모델 계열에서는 코드가 가장 짧지만, 루프 구조가 다른 연구에는 맞지 않는다.

어느 도구든 **기본값**이 많다. 검증 주기, 로깅 주기, gradient clipping 여부, 정밀도, checkpoint 저장 기준, 학습률 스케줄과 warm-up, 데이터 셔플 방식이 모두 기본값을 갖는다. 기본값은 대부분 합리적이지만, 그것이 **내 실험의 설정이 된다는 사실**은 변하지 않는다. PyTorch 자체에도 이런 기본값이 있다. optimizer를 Adam에서 AdamW로 바꾸면서 인자를 그대로 두면 무엇이 달라지는가.

```python {exec="python"}
for opt_cls in (torch.optim.Adam, torch.optim.AdamW):
    model, _, data = setup()
    opt = opt_cls(model.parameters(), lr=1e-2)                 # 다른 인자는 주지 않았다
    fit(model, opt, data, regression_step, max_epochs=200)
    norm = torch.cat([p.detach().flatten() for p in model.parameters()]).norm()
    loss = sum(regression_step(model, batch).item() for batch in data) / len(data)
    print(f"{opt_cls.__name__:5s} weight_decay 기본값 {opt.defaults['weight_decay']:<5}  "
          f"파라미터 노름 {norm:.4f}  학습 손실 {loss:.4f}")
```

```output
Adam  weight_decay 기본값 0      파라미터 노름 6.8874  학습 손실 0.0120
AdamW weight_decay 기본값 0.01   파라미터 노름 6.7403  학습 손실 0.0135
```

AdamW는 weight decay를 지정하지 않아도 0.01을 적용한다(집필 시점 예시. PyTorch 2.15에서 확인). "optimizer만 바꿨다"고 기록한 실험이 실제로는 regularization도 함께 바꾼 실험이 된다. 이 작은 문제에서는 차이가 작지만, 학습률이 크거나 학습이 길면 커진다(10.4장). 프레임워크의 기본값은 이런 항목이 수십 개다.

::quiz{id="adamw-default-weight-decay"}

## 도입 판단

프레임워크를 쓸지는 취향이 아니라 비용의 비교다. 프레임워크는 **반복 코드**를 줄여 주는 대신 **숨겨진 동작**을 늘린다. 판단 기준은 다음과 같다.

**도입하면 이득이 큰 경우**

- 여러 GPU·여러 노드, mixed precision, gradient accumulation처럼 **표준화되어 있고 틀리기 쉬운** 반복 코드가 필요할 때. 직접 쓴 분산 학습 코드의 버그는 찾기 어렵다(22.6장).
- 학습 루프 자체는 연구 대상이 아니고, 모델이나 데이터가 연구 대상일 때.
- 팀이 이미 그 도구로 일하고 있고, 코드를 서로 읽어야 할 때.

**직접 쓰는 편이 나은 경우**

- **루프의 구조가 연구 대상**일 때: 새 optimizer, 이중수준 최적화, 비표준 업데이트 순서, step마다 다른 데이터 소스를 섞는 학습. 프레임워크의 추상화를 우회하는 코드가 프레임워크 없이 쓴 코드보다 길어진다.
- 학습 루프를 처음 배우는 단계. 프레임워크가 대신하는 일을 모르면 프레임워크가 실패할 때 할 수 있는 것이 없다. 이 부가 루프를 직접 쓰는 것부터 시작한 이유다.
- 실험이 CPU나 GPU 하나에서 끝나는 작은 규모일 때. 이때 반복 코드는 100–200줄이고, 그 줄은 모두 내가 이해한다.

**도입한다면 지킬 것**

- 사용하는 버전을 고정하고(3.3장), 버전을 올릴 때 결과가 바뀌는지 확인한다.
- **해석이 끝난 설정 전체**, 즉 내가 지정하지 않은 기본값까지 포함한 설정을 실행마다 기록한다(6.3장). 대부분의 프레임워크는 이를 출력하거나 저장하는 방법을 제공한다.
- 직접 쓴 짧은 루프와 같은 결과를 내는지, 작은 문제에서 한 번 비교한다. 다르면 차이를 설명할 수 있을 때까지 넘어가지 않는다.
- 프레임워크 밖에서 돌아가는 `eval.py`를 유지한다. 학습은 프레임워크로 해도 평가는 프레임워크 없이 checkpoint만으로 재현할 수 있어야 한다.

::quiz{id="framework-adoption"}

:::note[분야 예시]
기후·기상 분야의 대형 예측 모델처럼 수백 개의 GPU로 학습하는 프로젝트는 분산 학습과 정밀도 관리가 핵심이라 프레임워크나 그에 준하는 공용 학습 코드를 쓰는 경우가 많다. 반면 물리 법칙을 손실에 넣는 연구(24.2장)는 손실 항마다 다른 데이터(경계 조건 점, 내부 collocation 점, 관측값)를 섞고 항의 가중치를 학습 중에 바꾸기도 한다. 이런 연구는 step 함수가 연구의 대부분이므로, 얇은 trainer에 step 함수를 넘기는 구조가 잘 맞는다. 같은 "딥러닝 학습"이라도 무엇이 연구 대상인지에 따라 답이 다르다.
:::

## 실습

직접 작성한 trainer로 실험을 구현하고, 프레임워크 없이 쓴 루프와 결과가 일치하는지, 중단·재개가 정확한지, 학습 코드 없이 평가할 수 있는지 확인한다. 코드는 `labs/deep-learning-pytorch/trainer-design/`에 있다.

- `trainer.py`: `Trainer`(루프·검증·callback·상태 저장)와 callback 세 개(`EarlyStopping`, `Checkpoint`, `PrintLogger`). gradient clipping과 scheduler는 **기본값이 '하지 않음'**이다. 지정하지 않은 동작이 일어나지 않게 하려는 선택이다.
- `task.py`: 과제 정의. 데이터, 모델, `regression_step`, 평가 함수. `trainer.py`를 import하지 않는다.
- `run.py`: (1) 직접 쓴 루프와 `Trainer`의 결과 비교, (2) callback을 붙인 학습과 `min_delta`에 따른 차이, (3) 15 epoch에서 저장·재개한 학습과 끊김 없는 학습의 비교.
- `eval.py`: checkpoint만 받아 다른 batch 크기로 다시 평가하고, checkpoint에 저장된 metric과 비교한다.

```bash
cd labs/deep-learning-pytorch/trainer-design
uv run --project ../.. --group torch python run.py
uv run --project ../.. --group torch python eval.py outputs/best.pt outputs/last.pt
```

집필 시점의 실행 결과(CPU, PyTorch 2.15)는 다음과 같다.

```output
$ python run.py
[1] 직접 쓴 루프와 Trainer, 20 epoch
    파라미터 일치: True   metric: {'val_rmse': 0.1261, 'val_mae': 0.0971}
[2] EarlyStopping + Checkpoint + PrintLogger
epoch  10  step   160  val_rmse 0.2373  val_mae 0.1939
epoch  20  step   320  val_rmse 0.1261  val_mae 0.0971
epoch  30  step   480  val_rmse 0.1068  val_mae 0.0824
epoch  40  step   640  val_rmse 0.1050  val_mae 0.0855
epoch  43  step   688  val_rmse 0.1052  val_mae 0.0862
    min_delta 0.0001: 멈춘 epoch 300  val_rmse 0.0614
    min_delta 0: 멈춘 epoch 300  val_rmse 0.0614
[3] 15 epoch에서 저장하고 새 Trainer로 재개, 끊김 없는 40 epoch와 비교
    파라미터 일치: True   early stopping 상태 일치: True
$ python eval.py outputs/best.pt outputs/last.pt
best.pt  epoch  36  다시 계산 {'val_rmse': 0.1035, 'val_mae': 0.0825}  저장된 값과 같은가 True
last.pt  epoch  43  다시 계산 {'val_rmse': 0.1052, 'val_mae': 0.0862}  저장된 값과 같은가 True
$ wc -l trainer.py task.py eval.py
 131 trainer.py
  41 task.py
  25 eval.py
 197 total
```

결과에서 읽을 것은 네 가지다.

- **[1]** trainer로 옮긴 루프가 직접 쓴 루프와 비트 단위로 같다.
- **[2]** early stopping의 `min_delta=1e-3`은 epoch 43에서 학습을 멈췄고(best는 epoch 36, val_rmse 0.1035), `min_delta`를 1e-4나 0으로 두면 300 epoch까지 학습해 val_rmse 0.0614에 이른다. 검증 곡선이 epoch당 0.001보다 천천히, 그러나 꾸준히 내려가는 구간을 "정체"로 판정한 것이다. 기본값 하나가 결론을 바꾼 예이며, 프레임워크의 early stopping을 쓸 때도 같은 일이 생긴다.
- **[3]** callback의 상태까지 저장했으므로 재개한 학습이 끊김 없는 학습과 같다.
- `eval.py`는 trainer 없이 batch 크기 7로 다시 계산해 저장된 metric을 재현한다. 코드량은 `trainer.py` 131줄, `task.py` 41줄, `eval.py` 25줄이다.

이어서 다음을 한다.

1. 연습문제로 early stopping 규칙을 구현한다.
2. 자신의 연구실 템플릿에 맞게 `trainer.py`와 `eval.py`를 고치고, 각 파일의 첫 docstring에 **책임과 책임이 아닌 것**을 적는다(6.7장).
3. (선택) PyTorch Lightning을 별도 환경에 설치할 수 있다면, `task.py`의 모델·데이터·step을 그대로 써서 같은 실험을 구현한다. 같은 epoch 수에서 `run.py`의 [1]과 같은 숫자가 나오는지 비교하고, 다르다면 어떤 기본값(검증 주기, 정밀도, clipping, 셔플, 진행 표시용 손실 평균 등)이 차이를 만들었는지 해석된 설정에서 찾는다. 코드량과, 루프 구조를 바꾸는 실험(예: step마다 두 데이터 소스 섞기)을 추가하는 비용을 함께 기록한다.

::exercise{id="early-stopping-rule"}

## 실패 사례

- **이해하지 못한 프레임워크 기본값이 결과를 바꾸는 것.** 검증 주기, 정밀도, clipping, early stopping 기준, 진행 표시줄의 손실이 어떤 평균인지를 모른 채 결과를 비교한다. 두 실험의 차이가 내가 바꾼 설정이 아니라 기본값이 바뀐 데서(예: 프레임워크 버전 업데이트) 올 수 있다. 해석된 설정 전체를 기록하지 않았다면 나중에 확인할 방법도 없다.
- **모든 것을 처리하는 거대한 trainer.** `if task == "classification"`, `if use_gan`, `if dataset == "climate"`가 trainer 안에 쌓인다. 한 실험을 위해 고친 줄이 다른 실험의 결과를 조용히 바꾸고, 어떤 플래그 조합이 실제로 테스트되었는지 아무도 모른다. 과제별 코드는 step 함수로, 부가 기능은 callback으로, 다른 구조의 루프는 다른 trainer로 뺀다.
- **trainer에 묶인 평가.** 평가가 trainer의 메서드라서 checkpoint를 평가하려면 optimizer와 학습 데이터 loader까지 만들어야 한다. 학습 코드가 바뀌면 옛 checkpoint를 평가할 수 없게 된다.
- **callback의 상태를 저장하지 않는 재개.** early stopping이나 best checkpoint의 기준값이 재개 때마다 초기화되어, 재개한 학습이 더 늦게 멈추거나 더 나쁜 checkpoint를 best로 덮어쓴다.

## 통과 기준

- 연구실 템플릿의 `trainer.py`와 `eval.py`를 설계하고, 각 파일에 책임·입출력·확장 지점(step 함수, callback 이벤트)을 문서화한다.
- trainer로 옮긴 학습이 직접 쓴 루프와 같은 결과를 내고, 중단·재개한 학습이 끊김 없는 학습과 같음을 테스트로 보인다.
- `eval.py`가 trainer 없이 checkpoint만으로 학습 중 기록한 metric을 재현한다.
- 연습문제 `early-stopping-rule`의 테스트를 통과한다.
