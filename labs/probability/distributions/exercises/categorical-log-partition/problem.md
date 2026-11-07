# categorical 분포의 log-partition

$K$개 범주의 categorical 분포를 지수족으로 쓰면, 자연 파라미터 $\boldsymbol{\eta} \in \mathbb{R}^K$(분류 모델의 logit)에 대해

$$
p(y = k \mid \boldsymbol{\eta}) = \exp\big(\eta_k - A(\boldsymbol{\eta})\big), \qquad
A(\boldsymbol{\eta}) = \log \sum_{j=1}^K e^{\eta_j}
$$

이다. $A$가 log-partition이다. 세 함수를 완성한다.

1. `log_partition(eta)`: $A(\boldsymbol{\eta})$. `eta`의 shape은 `(K,)`이고 스칼라를 반환한다. **`eta`의 성분이 1000처럼 커도 `inf`나 `nan` 없이** 계산해야 한다.
2. `mean_params(eta)`: 범주별 확률 $\pi_k = p(y = k \mid \boldsymbol{\eta})$, shape `(K,)`. 이것이 softmax다.
3. `log_prob(eta, k)`: $\log p(y = k \mid \boldsymbol{\eta})$. 큰 logit에서도 유한해야 한다.

테스트는 다음을 확인한다.

- 큰 logit에서의 수치 안정성.
- 모든 성분에 같은 상수를 더해도 확률이 바뀌지 않는다(자연 파라미터가 하나로 정해지지 않는 이유).
- $\nabla A(\boldsymbol{\eta}) = \boldsymbol{\pi}$, 즉 log-partition의 gradient가 충분통계량(one-hot 벡터)의 평균이다. 테스트는 유한차분으로 확인한다.

힌트: $\log\sum_j e^{\eta_j} = m + \log\sum_j e^{\eta_j - m}$이고, $m = \max_j \eta_j$로 두면 지수의 인자가 모두 0 이하가 된다(log-sum-exp 기법).
