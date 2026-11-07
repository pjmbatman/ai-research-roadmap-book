# gradient accumulation이 있는 학습 루프

메모리 때문에 큰 batch를 한 번에 올릴 수 없으면 작은 micro-batch 여러 개의 gradient를 모아 한 번 갱신한다(10.4장). 올바르게 구현하면 micro-batch 크기 8 × 누적 4회는 batch 크기 32와 **같은 학습**이어야 한다. 선형회귀 $\hat{y} = \mathbf{x}^\top\mathbf{w} + b$, 손실 $L = \frac{1}{n}\sum_i(\hat{y}_i - y_i)^2$의 학습 루프를 NumPy로 쓴다.

`train_linear(X, y, micro_batch, accum, lr, epochs, seed)`를 완성한다. 반환은 `(w, b, n_steps)`이고, `n_steps`는 파라미터를 갱신한 횟수(optimizer step 수)다.

규칙은 다음과 같다.

1. 파라미터는 `w = 0`(shape `(D,)`), `b = 0`에서 시작한다.
2. epoch `e`의 표본 순서는 `np.random.default_rng([seed, e]).permutation(N)`이다.
3. 그 순서를 앞에서부터 `micro_batch`개씩 자른다(마지막 micro-batch는 작을 수 있다).
4. micro-batch `accum`개마다, 그리고 epoch이 끝날 때 남은 micro-batch가 있으면, 모은 gradient로 SGD 한 step을 한다: $\theta \leftarrow \theta - \text{lr}\cdot g$.
5. $g$는 **그 step에 쓰인 모든 표본에 대한 평균 손실의 gradient**여야 한다. micro-batch마다 크기가 다를 수 있으므로, micro-batch 평균 gradient를 단순 평균하지 말고 표본 수로 가중한다(또는 표본별 gradient의 합을 모아 마지막에 총 표본 수로 나눈다).
6. gradient를 모으는 동안에는 파라미터를 바꾸지 않는다. step이 끝나면 모은 gradient를 비운다(`zero_grad`).

테스트는 (micro 8, accum 4)와 (micro 32, accum 1)의 결과가 같은지, step 수가 맞는지, 표본 수가 나누어떨어지지 않을 때도 같은지, 학습이 실제로 문제를 푸는지 확인한다.
