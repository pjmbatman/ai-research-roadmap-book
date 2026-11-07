---
id: experiment-operations/inference-and-deployment
title: "추론 최적화와 배포"
title_en: "Inference Optimization and Deployment"
tier: reference
status: draft
prerequisites: [deep-learning-pytorch/evaluation-and-checkpoints]
---

# 추론 최적화와 배포

학습이 끝난 모델은 다른 곳에서 쓰인다. 실험 장비의 제어 프로그램 안에서, 대량의 시뮬레이션 결과를 걸러 내는 파이프라인 안에서, 동료가 쓰는 웹 서비스 뒤에서 돈다. 이때 모델은 학습 코드에서 떨어져 나와 **다른 형식, 다른 실행 환경, 다른 정밀도**로 실행된다. 그 과정의 어느 단계에서든 모델이 내는 숫자가 조용히 바뀔 수 있다. 내보낸 모델이 원본과 같은 결과를 내는지 확인하지 않으면, 논문에 보고한 성능과 실제로 배포된 모델의 성능은 다른 것이 된다.

이 장은 학습된 모델을 내보내고(export), 추론 비용을 줄이고(양자화·pruning·distillation), 서빙하는 과정을 다룬다. 각 단계에서 무엇이 바뀔 수 있는지, 그것을 어떻게 측정하는지가 중심이다. 체크포인트의 내용과 inference contract는 17.6장, 지연·처리량 측정의 원칙은 [22.4장](../04-profiling-and-efficiency/index.md), 정수와 부동소수점 표현은 [5.5장](../../05-python-and-computing/05-computer-architecture/index.md)과 [9.1장](../../09-numerical-methods/01-floating-point/index.md)에서 다뤘다. 이 장의 도구 이름과 API는 빠르게 바뀌므로, 코드 예시는 모두 집필 시점 예시로 읽고 쓰는 버전의 문서에서 확인한다.

## 학습 목표

- 학습된 모델을 내보내고 원본과 같은 결과를 내는지 검증한다
- 추론 비용을 줄이는 기법의 효과와 오차를 측정한다
- batch 크기와 서빙 방식에 따른 지연·처리량을 설명한다

## 내보내기

**왜 필요한가.** 학습에 쓴 PyTorch 코드는 Python 인터프리터, 모델 클래스의 소스 코드, 그 버전의 라이브러리를 모두 필요로 한다. 배포 환경에는 이것들이 없거나 다를 수 있다. Python이 없는 C++ 프로그램, 특정 하드웨어용 추론 엔진, 다른 언어의 서비스가 모델을 실행해야 할 때는 **모델의 계산을 코드와 분리된 형식**으로 내보내야 한다.

**정의.** 내보내기(export)는 모델의 forward 계산을 연산 그래프(어떤 연산을 어떤 순서로, 어떤 텐서에 적용하는가)와 가중치로 바꿔 저장하는 일이다. 집필 시점 PyTorch의 주요 경로는 다음과 같다.

- **`torch.export`**: 예시 입력으로 forward를 추적해 연산 그래프(`ExportedProgram`)를 만든다. 저장(`torch.export.save`)·불러오기(`torch.export.load`)를 할 수 있고, 이 그래프를 다른 도구가 받아 최적화·컴파일한다.
- **ONNX**: 여러 프레임워크와 추론 엔진이 공유하는 그래프 형식이다. `torch.onnx.export`로 만들고 ONNX Runtime 같은 엔진으로 실행한다. `onnx` 등 별도 패키지가 필요하다.
- **TorchScript**(`torch.jit.trace`·`torch.jit.script`): 이전 세대의 방식이다. 집필 시점 PyTorch 문서는 이를 deprecated로 표시하고 있으므로 새 프로젝트에서는 위 두 경로를 쓴다.

**직관.** 내보내기는 **예시 입력 하나로 계산을 녹화**하는 것에 가깝다. 그래서 녹화할 때의 상태가 그대로 그래프에 들어간다. 모델이 train 모드였으면 dropout과 BatchNorm의 batch 통계가 들어가고, 예시 입력의 batch 크기를 고정된 것으로 녹화하면 다른 batch 크기를 받지 못한다. 입력값에 따라 갈라지는 Python `if` 문은 하나의 그래프로 녹화할 수 없어 `torch.export`는 오류를 낸다.

**전처리 포함 여부.** 정규화·단위 변환·토큰화 같은 전처리를 그래프 안에 넣을지 밖에 둘지 정해야 한다. 그래프 안에 넣으면 배포 환경이 전처리를 다시 구현할 필요가 없어 불일치가 줄어든다. 밖에 두면 전처리를 바꾸기 쉽지만, 학습 때와 똑같이 구현했는지를 따로 검증해야 한다("추론 계약의 유지"). 어느 쪽이든 **내보낸 모델의 입력이 정확히 무엇인지**(shape, dtype, 단위, 정규화 여부)를 문서로 남긴다.

```python {exec="python"}
import io
import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
for _ in range(5):
    model(torch.randn(16, 4))                      # train 모드 forward: BatchNorm 이동 평균이 갱신된다
model.eval()                                       # 내보내기 전에 평가 모드로

x = torch.randn(8, 4)                              # 예시 입력, (B, 4)
batch = torch.export.Dim("batch")                  # batch 축은 크기가 바뀔 수 있다고 알린다
ep = torch.export.export(model, (x,), dynamic_shapes=({0: batch},))
buf = io.BytesIO()
torch.export.save(ep, buf)                         # 파일 대신 메모리 버퍼에 저장
buf.seek(0)
deployed = torch.export.load(buf).module()

with torch.no_grad():
    for n in [8, 3, 100]:                          # 예시와 다른 batch 크기도 확인
        xt = torch.randn(n, 4)
        diff = (deployed(xt) - model(xt)).abs().max().item()
        print(f"batch {n:3d}: 최대 차이 {diff:.1e}  allclose: {torch.allclose(deployed(xt), model(xt), atol=1e-6)}")
```

```output
batch   8: 최대 차이 0.0e+00  allclose: True
batch   3: 최대 차이 0.0e+00  allclose: True
batch 100: 최대 차이 0.0e+00  allclose: True
```

평가 모드로 내보내고 batch 축을 동적(`Dim("batch")`)으로 지정했으므로, 저장·불러오기를 거친 모델이 예시와 다른 batch 크기에서도 원본과 같은 값을 낸다. 같은 기계, 같은 연산 순서이므로 차이가 정확히 0이다. 다른 엔진이나 하드웨어에서 실행하면 연산 순서와 kernel이 달라져 작은 차이가 생기는 것이 정상이고, 그때는 허용 오차를 정해 `allclose`로 판정한다.

평가 모드 전환을 빠뜨리면 어떻게 되는지 본다.

```python {exec="python"}
model.train()
ep_train = torch.export.export(model, (x,))       # eval()을 빠뜨렸다
model.eval()
with torch.no_grad():
    a, b = ep_train.module()(x), model(x)
print("train 모드로 내보낸 모델과 평가 모드 모델의 최대 차이 > 1e-3:", (a - b).abs().max().item() > 1e-3)
```

```output
train 모드로 내보낸 모델과 평가 모드 모델의 최대 차이 > 1e-3: True
```

train 모드로 녹화된 그래프는 BatchNorm을 batch 통계로 계산하고 dropout을 적용한다. 오류나 경고 없이 내보내지고, 출력의 shape도 맞다. 숫자를 원본과 비교해야만 드러난다.

::quiz{id="export-eval-mode"}

ONNX로 내보내는 과정도 같은 절차를 따른다. 아래는 이 환경에 `onnx` 패키지가 없어 실행하지 않은 집필 시점 예시다.

```python
# 집필 시점 예시: onnx, onnxruntime 패키지가 필요하다
import numpy as np
import onnxruntime as ort

model.eval()
torch.onnx.export(model, (x,), "model.onnx", input_names=["x"], output_names=["y"],
                  dynamic_shapes=({0: torch.export.Dim("batch")},))
sess = ort.InferenceSession("model.onnx")
y_onnx = sess.run(None, {"x": x.numpy()})[0]
print(np.abs(y_onnx - model(x).detach().numpy()).max())   # 원본과 비교한다
```

**연구에서의 의미.** 내보낸 파일은 그 자체로 하나의 결과물이다. 파일과 함께 다음을 기록한다. 원본 체크포인트(경로·해시), 내보낼 때의 PyTorch 버전과 내보내기 방식, 입력 명세(shape·dtype·단위·정규화), 그리고 고정된 검증 입력에 대한 원본의 출력. 마지막 항목이 있으면 배포 환경에서 언제든 같은 입력을 넣어 출력이 같은지 확인할 수 있다.

## 추론 최적화

추론 비용은 지연(요청 하나에 걸리는 시간), 처리량(초당 처리하는 표본 수), 메모리, 전력으로 잰다. 이를 줄이는 기법은 크게 세 가지다. 모두 **모델이 내는 숫자를 바꾸므로**, 비용과 함께 정확도 변화를 측정해 보고한다.

**양자화(quantization).** 가중치(와 활성값)를 int8 같은 작은 정수로 표현한다. 메모리가 fp32의 4분의 1이 되고, 정수 연산을 지원하는 하드웨어에서는 계산도 빨라진다. 대칭 양자화는 0을 0으로 보내고 절댓값 최댓값을 127에 맞춘다.

$$
s = \frac{\max_i \lvert w_i \rvert}{127}, \qquad q_i = \operatorname{clip}\big(\operatorname{round}(w_i / s), -127, 127\big), \qquad \hat{w}_i = s\,q_i.
$$

$s$는 scale, $q_i$는 저장하는 int8 값, $\hat{w}_i$는 복원값이다. 반올림 오차는 원소마다 $s/2$ 이하다. 핵심 선택은 **scale을 얼마나 잘게 두느냐**다. 텐서 전체에 scale 하나를 쓰면(per-tensor) 크기가 큰 행이 scale을 정해 작은 행의 정밀도를 빼앗는다. 출력 채널(가중치 행렬의 행)마다 scale을 두면(per-channel) 이 문제가 준다.

```python {exec="browser" session="infer"}
import numpy as np

rng = np.random.default_rng(0)
row_scale = np.array([0.01, 0.1, 1.0, 5.0])[:, None]           # 출력 채널마다 크기가 다르다
W = rng.normal(size=(4, 64)) * row_scale                       # (out, in) 가중치
x = rng.normal(size=64)                                        # (in,) 입력

def quant_sym(w, axis=None):
    amax = np.abs(w).max(axis=axis, keepdims=axis is not None)
    scale = amax / 127                                         # 최댓값이 ±127로 간다
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale

for name, axis in [("텐서 하나", None), ("행별", 1)]:
    q, s = quant_sym(W, axis)
    W_hat = q * s                                              # 복원(dequantize)
    rel_row = np.linalg.norm(W_hat - W, axis=1) / np.linalg.norm(W, axis=1)
    rel_out = np.abs(W_hat @ x - W @ x) / np.abs(W @ x)
    print(f"{name:6s} 가중치 상대오차 {np.round(rel_row, 4)}  출력 상대오차 {np.round(rel_out, 4)}")
print(f"메모리: fp32 {W.size * 4} byte → int8 {W.size} byte + scale")
```

```output
텐서 하나  가중치 상대오차 [1.     0.3689 0.0374 0.0061]  출력 상대오차 [1.     0.2509 0.015  0.0396]
행별     가중치 상대오차 [0.0055 0.0054 0.0058 0.0061]  출력 상대오차 [0.003  0.0085 0.0006 0.0396]
메모리: fp32 1024 byte → int8 256 byte + scale
```

텐서 하나의 scale을 쓰면 크기 0.01인 첫 행은 모든 원소가 $s/2$보다 작아 **전부 0**이 되고(상대오차 1), 그 출력 채널은 아무것도 계산하지 않는다. 행별 scale에서는 모든 행의 가중치 오차가 0.6% 안팎으로 고르다. 마지막 행은 두 방식에서 scale이 같아 결과도 같다. 출력 상대오차가 가중치 오차보다 큰 행(마지막 행 4%)이 있는 것은, 내적의 항들이 서로 상쇄되어 출력값 자체가 작아지면 같은 절대 오차가 큰 상대 오차가 되기 때문이다. 그래서 양자화의 영향은 가중치의 오차가 아니라 **모델 출력과 최종 metric**으로 판단한다.

::quiz{id="per-channel-scale"}

정수 연산으로 빨라지려면 입력도 정수로 바꾸고, 곱셈과 덧셈을 정수로 한 뒤 마지막에 한 번만 실수로 되돌린다. $y_j = \sum_k W_{jk} x_k \approx s_{W,j}\, s_x \sum_k q^{W}_{jk} q^{x}_k$이므로, 정수 곱의 합에 scale 두 개를 곱하면 된다.

```python {exec="browser" session="infer"}
qW, sW = quant_sym(W, 1)                                       # (4, 64) int8, (4, 1)
qx, sx = quant_sym(x)                                          # (64,) int8, 스칼라
acc32 = qW.astype(np.int32) @ qx.astype(np.int32)              # 정수 곱의 합, (4,)
y_int8 = acc32 * sW[:, 0] * sx                                 # 실수로 되돌린다
print("float 출력 :", np.round(W @ x, 4))
print("int8 출력  :", np.round(y_int8, 4))
worst = 64 * 127 * 127                                         # 길이 64 내적의 최악 누적값
print("최악의 누적값:", worst, " int16 최댓값:", np.iinfo(np.int16).max, " int32로 안전한 최대 길이:",
      np.iinfo(np.int32).max // (127 * 127))
```

```output
float 출력 : [-0.0739  0.5519  8.5956  5.0563]
int8 출력  : [-0.0739  0.5305  8.7069  5.0275]
최악의 누적값: 1032256  int16 최댓값: 32767  int32로 안전한 최대 길이: 133144
```

입력까지 양자화하자 오차가 커졌다(세 번째 출력이 1.3% 다르다). 입력 $x$도 텐서 하나의 scale을 썼기 때문이다. 활성값은 입력마다 범위가 달라서, 실제 양자화에서는 대표 입력 몇백 개로 활성값의 범위를 미리 재는 **calibration** 단계를 거쳐 scale을 정한다. 마지막 줄은 정수 하드웨어가 int8 곱을 **int32로 누적**하는 이유다. int8 곱 하나는 최대 $127^2 = 16129$이므로 길이 64의 내적만으로 int16 범위를 크게 넘을 수 있고, int32는 길이 13만 개까지 안전하다. 5.5장에서 본 것처럼 고정 폭 정수는 넘쳐도 오류 없이 감긴다.

**pruning.** 크기가 작은 가중치를 0으로 만들어 계산과 저장을 줄인다. 그러나 0이 흩어져 있는(비정형) 희소 행렬은 일반 GPU의 밀집 행렬 곱에서 빨라지지 않는다. 속도를 얻으려면 행·채널 단위로 통째로 지우거나(정형 pruning), 하드웨어가 지원하는 희소 패턴을 따라야 한다. pruning 뒤에는 보통 짧은 재학습(fine-tuning)으로 정확도를 회복한다.

**distillation.** 큰 모델(teacher)의 출력을 정답 대신 또는 정답과 함께 목표로 삼아 작은 모델(student)을 학습시킨다. teacher의 출력 분포는 클래스 사이의 유사성 같은 정보를 담고 있어, 같은 데이터로 student를 처음부터 학습할 때보다 좋은 결과를 내는 경우가 많다. 과학 문제에서는 비싼 시뮬레이터나 큰 모델의 예측을 대량으로 만들어 작은 대리 모델(surrogate)을 학습하는 것도 같은 구조다. student가 teacher의 오류까지 배운다는 점, 그리고 teacher를 만든 데이터 범위 밖에서는 둘 다 검증되지 않았다는 점을 기억한다.

| 기법 | 줄이는 것 | 대가 | 확인할 것 |
|---|---|---|---|
| 양자화 | 메모리, (정수 연산 하드웨어에서) 지연 | 반올림 오차 | 출력 차이, 최종 metric, calibration 데이터 |
| pruning | 파라미터 수, (정형일 때) 계산 | 정확도, 재학습 비용 | 실제 지연이 줄었는가 |
| distillation | 모델 크기 전체 | 학습 비용, teacher의 편향 상속 | teacher와의 차이, 분포 밖 성능 |

:::note[분야 예시]
천문 관측이나 입자 검출기에서는 초당 쏟아지는 신호를 실시간으로 걸러야 해서, 학습한 모델을 FPGA 같은 저전력 장치에 올리는 연구가 있다. 이런 장치에서는 정수 연산과 작은 모델이 필수라 양자화와 distillation이 기본 단계가 된다. 이때 검증 기준은 "출력이 원본과 같은가"가 아니라 "물리적으로 의미 있는 사건을 원본만큼 놓치지 않는가"다. 최적화의 효과는 과학적 목적에 맞는 metric으로 측정한다.
:::

## 서빙

**batch 추론과 온라인 추론.** 추론은 두 방식으로 나뉜다.

- **batch 추론**(offline): 쌓여 있는 데이터 전체에 모델을 적용한다. 시뮬레이션 결과 수백만 개를 거르거나, 위성 영상 전체를 분류하는 일이다. 목표는 **처리량**이고, 요청 하나의 지연은 중요하지 않다. 큰 batch로 GPU를 꽉 채운다.
- **온라인 추론**(online): 요청이 올 때마다 바로 답한다. 실험 장비가 측정값을 보내면 다음 조건을 정해 주는 경우다. 목표는 **지연**, 특히 느린 쪽 꼬리(예: 99번째 백분위수 지연)다.

**batch 크기의 거래.** GPU 계산에는 batch 크기와 무관한 고정 비용(kernel launch, 데이터 전송, 프레임워크 오버헤드)이 있어서, batch를 키우면 처리량이 오른다(5.5장의 연산 강도). 온라인 서빙에서는 요청을 잠깐 모아 batch로 처리하는 **동적 batching**(dynamic batching)으로 처리량을 올리는데, 모으는 동안 요청이 기다리므로 지연이 는다. 단순한 비용 모형으로 이 거래를 계산한다. batch 하나의 계산 시간을 $t(B) = t_0 + B\,c$, 요청이 초당 $\lambda$개 고르게 도착한다고 하면, batch를 채우는 동안 요청은 평균 $(B-1)/(2\lambda)$를 기다린다.

```python {exec="browser" session="infer"}
t_fixed_ms, t_per_sample_ms = 4.0, 0.05            # batch마다 고정 비용, 표본당 비용 (가정)
arrival_per_s = 2000                               # 초당 요청 수 (가정)
print("batch  계산 지연(ms)  모으는 대기(ms)  총 지연(ms)  처리량(표본/s)")
for B in [1, 8, 32, 128, 512]:
    compute_ms = t_fixed_ms + B * t_per_sample_ms
    wait_ms = (B - 1) / arrival_per_s * 1000 / 2   # batch를 채우는 동안 요청이 평균적으로 기다리는 시간
    print(f"{B:5d} {compute_ms:13.2f} {wait_ms:15.2f} {compute_ms + wait_ms:12.2f} {B / compute_ms * 1000:15.0f}")
```

```output
batch  계산 지연(ms)  모으는 대기(ms)  총 지연(ms)  처리량(표본/s)
    1          4.05            0.00         4.05             247
    8          4.40            1.75         6.15            1818
   32          5.60            7.75        13.35            5714
  128         10.40           31.75        42.15           12308
  512         29.60          127.75       157.35           17297
```

비용 상수는 가정이지만 모양은 일반적이다. batch 1의 처리량은 초당 247개로, 초당 2000개 요청을 감당하지 못한다(요청이 쌓여 대기열이 끝없이 길어진다). batch 8만 되어도 처리량은 7배가 되고 지연은 2 ms 늘 뿐이다. batch 512에서는 처리량이 거의 포화(고정 비용이 묻힌다)하는데 지연은 39배가 된다. 그래서 서빙에서는 "지연 상한(예: 50 ms) 안에서 처리량을 최대화하는 batch"를 고른다. 이 모형에서는 batch 128이 그 답이다. 실제 상수는 측정해서 넣는다(22.4장의 원칙: warm-up 뒤 여러 번, GPU 동기화 포함, 평균과 꼬리 백분위수).

::widget{name="batching-latency"}

::quiz{id="batch-latency-throughput"}

**서빙 도구.** 집필 시점에는 모델을 HTTP·gRPC 요청으로 받는 추론 서버(예: NVIDIA Triton Inference Server, TorchServe, 대형 언어 모델 전용 서버들)와, 직접 쓰는 간단한 웹 서버(FastAPI 등) 위에 모델을 올리는 방법이 함께 쓰인다. 도구마다 동적 batching, 여러 모델 관리, 모니터링 기능이 다르고, 유지 관리 상태도 바뀌므로 선택하기 전에 공식 문서와 저장소의 현재 상태를 확인한다. 연구실 규모에서는 대개 batch 추론 스크립트 하나나 단순한 웹 서버로 충분하다.

**임베디드·엣지 환경.** 실험 장비 옆의 소형 컴퓨터, 현장 센서, 휴대 장치에서는 GPU가 없거나 작고, 메모리와 전력이 제한된다. 여기서는 내보낸 그래프를 그 장치용 런타임(ONNX Runtime 등)으로 실행하고, int8 양자화와 작은 모델이 사실상 필수다. 장치의 정밀도·지원 연산이 학습 환경과 다르므로, **그 장치에서** 검증 입력의 출력을 원본과 비교한다.

## 추론 계약의 유지

**왜 필요한가.** 17.6장의 inference contract는 모델이 받는 입력과 내는 출력의 약속이다. 입력의 shape·dtype·단위·정규화, 출력의 의미와 단위, 그리고 전처리·후처리의 정확한 정의가 여기에 들어간다. 배포에서 가장 흔한 오류는 모델이 아니라 **이 계약이 깨지는 것**이다. 모델 파일은 올바른데 전처리가 학습 때와 달라 결과가 틀린다.

**학습과 추론의 전처리 일치.** 정규화 상수(평균·표준편차)는 학습 데이터로 정한 **상수**이고 모델의 일부다. 추론 때 들어온 batch로 다시 계산하면 안 된다. 단위도 계약의 일부다.

```python {exec="browser" session="infer"}
rng = np.random.default_rng(1)
temp_c = rng.uniform(10, 40, size=200)             # 학습 데이터: 온도 (°C)
y_obs = 0.8 * temp_c + 3 + rng.normal(0, 0.5, size=200)
mu, sd = temp_c.mean(), temp_c.std()               # 학습 데이터로 정한 정규화 상수 — 계약의 일부
z = (temp_c - mu) / sd
a, b = np.polyfit(z, y_obs, 1)                     # 정규화된 입력에 대한 선형 모델

def predict(t, mean, std):
    return a * (t - mean) / std + b

new_c = np.array([15.0, 20.0, 25.0])               # 배포 후 들어온 요청 3개 (°C)
ref = predict(new_c, mu, sd)
print("학습 때의 상수      :", np.round(ref, 2))
print("요청 batch로 정규화 :", np.round(predict(new_c, new_c.mean(), new_c.std()), 2))
print("단위가 K로 들어옴   :", np.round(predict(new_c + 273.15, mu, sd), 2))
```

```output
학습 때의 상수      : [14.98 18.97 22.96]
요청 batch로 정규화 : [14.97 23.15 31.33]
단위가 K로 들어옴   : [232.88 236.86 240.85]
```

모델은 세 경우 모두 같다. 요청 batch의 평균·표준편차로 정규화하면 15–25 °C 요청이 학습 데이터의 넓은 범위(10–40 °C)처럼 늘어나 예측이 최대 8 이상 틀린다. 게다가 같은 요청이라도 함께 들어온 다른 요청에 따라 답이 바뀐다. 켈빈으로 들어온 입력은 학습 범위에서 수백 도 떨어진 외삽이 되어 터무니없는 값을 낸다. 둘 다 오류도 경고도 없다.

계약을 지키는 실무 규칙은 다음과 같다.

- 정규화 상수, 어휘 사전, 단위 변환 같은 전처리 상태를 **모델 체크포인트와 함께 저장**하고 함께 버전을 매긴다. 가능하면 내보낸 그래프 안에 넣는다.
- 입력 명세를 코드로 검사한다. shape·dtype·값의 범위(예: 온도가 학습 범위의 몇 배 밖이면 경고)를 서빙 코드의 입구에서 확인한다.
- **고정된 검증 입력과 기대 출력**을 모델과 함께 둔다. 배포할 때마다, 그리고 라이브러리를 올릴 때마다 이 입력으로 출력을 비교하는 테스트를 돌린다. 허용 오차는 미리 정해 둔다(예: fp32 엔진 간 비교는 상대 $10^{-5}$ 수준, int8 양자화는 최종 metric 기준).

::quiz{id="contract-normalization"}

## 실습

모델을 내보내고 원본과 출력 일치·속도를 비교한다. 코드는 `labs/experiment-operations/inference-and-deployment/export_check.py`다. 작은 MLP 회귀 모델을 학습하고, `torch.export`로 batch 축이 동적인 그래프를 파일(`outputs/model.pt2`)로 저장한 뒤 다시 불러온다. 그다음 학습 분포 안 $[-1, 1]^6$과 밖 $[-5, 5]^6$의 입력 각 1만 개로 원본과 출력을 비교하고, batch 크기별 지연을 잰다. 이 환경에는 `onnx` 패키지가 없어 ONNX 대신 `torch.export`를 썼다. ONNX를 쓸 수 있으면 앞의 집필 시점 예시로 같은 비교를 한다. 집필 시점 CPU(스레드 1개)에서의 실행 결과는 다음과 같다. 시간 값은 기계마다, 실행마다 다르다.

```output
내보낸 파일 model.pt2: 89 KiB
분포 안 [-1, 1]: 최대 절대 차이 0.00e+00, allclose(rtol=1e-5, atol=1e-6) True
분포 밖 [-5, 5]: 최대 절대 차이 0.00e+00, allclose(rtol=1e-5, atol=1e-6) True
batch   원본(ms)  내보낸 모델(ms)  처리량(표본/s, 내보낸 모델)
    1      0.040            0.099                10138
   32      0.060            0.121               264037
 1024      0.446            0.643              1592089
```

결과를 다음 순서로 읽는다.

1. **수치 일치는 분포 밖 입력까지 확인한다.** 정규화나 활성화 함수의 구현 차이는 큰 입력에서 먼저 드러난다. 여기서는 두 범위 모두 차이가 0이다.
2. **내보내기 자체는 빠르게 만들지 않는다.** 불러온 그래프를 Python에서 그대로 실행하면 원본보다 오히려 느렸다. 내보낸 그래프는 최적화의 **입력**이다. 속도는 그 그래프를 받는 컴파일러·추론 엔진(집필 시점 PyTorch의 AOTInductor, ONNX Runtime, 하드웨어 제조사의 엔진 등)에서 나온다. 엔진을 바꾸면 수치 비교(1번)를 다시 한다.
3. **batch를 키우면 처리량이 크게 는다.** batch 1에서 1024로 가면 지연은 몇 배 늘지만 처리량은 100배 이상 늘었다. 앞의 비용 모형에서 본 고정 비용의 효과다.

이어서 대칭 int8 양자화를 직접 구현한다. 테스트가 바로 채점한다.

::exercise{id="symmetric-int8"}

## 실패 사례

- **내보낸 모델의 수치 결과를 검증하지 않는 것.** 내보내기가 오류 없이 끝나고 출력 shape이 맞으면 성공한 것처럼 보인다. 그러나 train 모드로 녹화했거나, batch 크기가 고정되었거나, 전처리가 빠졌거나, 엔진의 정밀도가 다르면 숫자가 바뀐다. 고정된 검증 입력으로 원본과 비교하고, 그 결과를 내보낸 파일과 함께 남긴다.
- **양자화 오차를 가중치 수준에서만 보는 것.** 가중치 오차가 1%라도 상쇄가 큰 출력이나 범위가 넓은 활성값에서는 오차가 커진다. 최종 metric과, 가능하면 분포 밖 입력에서의 성능까지 원본과 비교한다.
- **추론 때 정규화 상수를 다시 계산하는 것.** 같은 입력이 함께 들어온 다른 요청에 따라 다른 답을 받는다. 정규화 상수는 학습 데이터로 정해 모델과 함께 저장한 상수다.
- **처리량만 보고 서빙 batch를 고르는 것.** 큰 batch는 처리량을 올리지만 요청이 batch를 채우기를 기다리는 시간이 지연에 더해진다. 지연 상한을 먼저 정하고 그 안에서 고른다.

## 통과 기준

- 학습한 모델을 내보낸 뒤 고정된 검증 입력(학습 분포 안과 밖)으로 원본과 출력을 비교해, 정한 허용 오차 안에서 같음을 보고한다.
- 양자화 등 추론 최적화를 하나 적용하고, 메모리·지연의 개선과 최종 metric의 변화를 함께 측정해 보고한다.
- 주어진 지연 상한에서 서빙 batch 크기를 비용 모형과 측정으로 정하고 근거를 설명한다.
- 연습문제 `symmetric-int8`의 테스트를 통과한다.

## 참고 문헌

- PyTorch 공식 문서 — `torch.export`(ExportedProgram, 동적 shape, 저장·불러오기), `torch.onnx`, AOTInductor. pytorch.org.
- ONNX와 ONNX Runtime 공식 문서. onnx.ai, onnxruntime.ai.
