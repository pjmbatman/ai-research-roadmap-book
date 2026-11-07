# BatchNorm의 train/eval 동작 구현하기

BatchNorm은 train 모드와 eval 모드에서 다른 통계를 쓴다. 함수 `batchnorm_forward`를 완성한다.

```python
y, new_mean, new_var = batchnorm_forward(x, gamma, beta, running_mean, running_var,
                                         training, momentum=0.1, eps=1e-5)
```

| 인자 | shape | 의미 |
|---|---|---|
| `x` | `(B, D)` | 입력 batch |
| `gamma`, `beta` | `(D,)` | 학습되는 scale과 shift |
| `running_mean`, `running_var` | `(D,)` | 지금까지 누적한 이동 평균 통계(buffer) |
| `training` | bool | train 모드 여부 |

**train 모드** (`training=True`)

1. 특성(열)마다 batch 평균 $\mu_B$와 **편향** 분산 $\sigma_B^2 = \frac{1}{B}\sum_i (x_i - \mu_B)^2$을 구한다.
2. $\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, $y = \gamma\hat{x} + \beta$로 정규화한다.
3. 이동 평균을 갱신한다. 평균은 $\mu_B$로, 분산은 **불편** 분산 $\frac{1}{B-1}\sum_i (x_i - \mu_B)^2$으로 갱신한다(PyTorch의 `BatchNorm1d`와 같은 규칙).

$$
\text{new} = (1 - m)\cdot\text{running} + m\cdot\text{batch 통계}
$$

4. `B == 1`이면 분산을 추정할 수 없으므로 `ValueError`를 낸다.

**eval 모드** (`training=False`)

- batch 통계 대신 `running_mean`, `running_var`로 정규화하고, 통계는 갱신하지 않는다(받은 값을 그대로 반환한다).
- 표본이 하나여도 동작해야 한다.

입력 배열을 제자리에서 바꾸지 않는다. 테스트는 수식, 이동 평균 갱신 규칙, eval 출력이 batch 구성과 무관한지, 입력을 바꾸지 않는지 확인한다.
