# MLP 학습 한 step: forward → loss → backward → update

본문의 작은 MLP를 학습하는 한 step을 직접 쓴다. 함수 `train_step(params, X, y, lr)`를 완성해 **갱신된 파라미터**와 **갱신 전 파라미터에서의 loss**를 반환한다.

모델은 은닉층 하나(tanh)의 MLP다.

$$
\mathbf{H} = \tanh(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1), \qquad
\hat{\mathbf{y}} = \mathbf{H}\mathbf{W}_2 + \mathbf{b}_2, \qquad
L = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2.
$$

| 이름 | shape |
|---|---|
| `X` | `(N, D)` |
| `y` | `(N, 1)` |
| `params["W1"]`, `params["b1"]` | `(D, H)`, `(H,)` |
| `params["W2"]`, `params["b2"]` | `(H, 1)`, `(1,)` |

순서는 다음과 같다.

1. **forward**: $\mathbf{H}$와 $\hat{\mathbf{y}}$를 계산한다.
2. **loss**: 위의 MSE $L$을 계산한다.
3. **backward**: 연쇄법칙으로 네 파라미터의 gradient를 구한다. $\partial L / \partial \hat{\mathbf{y}} = 2(\hat{\mathbf{y}} - \mathbf{y})/N$이고, $\tanh'(a) = 1 - \tanh^2(a)$다.
4. **update**: 각 파라미터 $p$를 $p - \text{lr} \cdot \partial L / \partial p$로 바꾼다.

반환값은 `(new_params, loss)`이고, `new_params`는 같은 키를 가진 **새** dict다. 입력으로 받은 `params`의 배열을 제자리에서 바꾸지 않는다.

테스트는 loss 값, gradient(중심 차분과 비교), 반복할 때 loss가 줄어드는지, 입력을 바꾸지 않는지를 확인한다. gradient는 `(params - new_params) / lr`로 복원해 비교한다.
