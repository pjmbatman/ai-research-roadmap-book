# 학습 메모리 예산 계산기

ReLU MLP를 학습할 때 GPU 메모리에 올라가는 네 가지를 byte로 센다. 층 폭이 `widths = [d_0, d_1, ..., d_L]`인 MLP는 `Linear(d_0, d_1) → ReLU → ... → Linear(d_{L-1}, d_L)`이다(마지막 Linear 뒤에는 ReLU가 없다).

| 항목 | 계산 |
|---|---|
| 파라미터 수 $P$ | 층마다 가중치 $d_{i}d_{i+1}$ + bias $d_{i+1}$ |
| `params` | $P \times$ (파라미터 byte) |
| `grads` | $P \times$ (gradient byte) |
| `optimizer` | $P \times$ (optimizer 상태 byte) |
| `activations` | batch $\times\ (d_0 + d_1 + \cdots + d_{L-1}) \times$ (활성값 byte) |
| `total` | 위 네 항목의 합 |

활성값 규칙은 본문 cell 2에서 측정한 것이다. 각 Linear는 backward를 위해 입력을 저장하고, ReLU가 저장하는 출력은 다음 Linear의 입력과 같은 텐서라 한 번만 센다. 원소당 byte는 `RECIPES`에 주어져 있다.

```python
RECIPES = {   # (파라미터, gradient, optimizer 상태, 활성값) 원소당 byte
    "fp32-sgd":  (4, 4, 0, 4),
    "fp32-adam": (4, 4, 8, 4),
    "bf16-adam": (2, 2, 12, 2),   # bf16 가중치·gradient, fp32 사본·m·v, bf16 활성값
}
```

두 함수를 완성한다.

1. `training_memory(widths, batch, recipe="fp32-adam")` — 키가 `params`, `grads`, `optimizer`, `activations`, `total`인 dict를 반환한다. 값은 byte 단위 `int`다.
2. `max_batch(widths, budget_bytes, recipe="fp32-adam")` — `total`이 `budget_bytes` 이하가 되는 가장 큰 batch를 반환한다. batch 1도 들어가지 않으면 0을 반환한다.

예: `widths = [2, 3, 1]`, batch 4, `"fp32-adam"`이면 파라미터는 $2 \cdot 3 + 3 + 3 \cdot 1 + 1 = 13$개이고, `params` 52, `grads` 52, `optimizer` 104, `activations` $4 \times (2 + 3) \times 4 = 80$, `total` 288 byte다.

힌트: 활성값만 batch에 비례하고 나머지는 batch와 무관하다. `max_batch`는 반복문 없이 나눗셈 한 번으로 구할 수 있다.

이 계산기는 출발점이다. 실제 학습에서는 손실 계산이 저장하는 텐서, CUDA context, allocator의 여유 공간이 더해지므로, 계산한 최대 batch를 그대로 쓰지 말고 측정으로 확인한다.
