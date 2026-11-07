# permutation importance 직접 구현하기

permutation importance는 학습이 끝난 모델을 고정하고, 평가 데이터에서 한 특징의 값만 무작위로 섞었을 때 오차가 얼마나 커지는지로 그 특징의 중요도를 잰다.

함수 `permutation_importance(predict, X, y, n_repeats=5, seed=0)`를 완성해 shape `(p,)`의 배열을 반환한다. `predict`는 `(N, p)` 배열을 받아 `(N,)` 예측을 내는 함수다.

1. 기준 오차 $E_0 = \mathrm{MSE}(y, \texttt{predict}(X))$를 계산한다.
2. 특징 $j = 0, \dots, p-1$마다 `n_repeats`번 반복한다. `X`의 **복사본**을 만들고 $j$번째 열을 `rng.permutation`으로 섞은 뒤 MSE $E_{j,r}$를 잰다.
3. 중요도는 $\frac{1}{R}\sum_r (E_{j,r} - E_0)$, 즉 섞었을 때 평균적으로 늘어난 MSE다.

난수는 `rng = np.random.default_rng(seed)` 하나로 만든다. 입력 `X`를 직접 바꾸면 안 된다.

테스트는 다음을 확인한다.

- 모델이 쓰지 않는 특징의 중요도는 정확히 0이다.
- $\hat y = 2x_0$인 모델에서 $x_0$의 중요도는 약 $8\operatorname{Var}(x_0)$이다. 섞인 $x_0'$이 원래 $x_0$와 독립이면 $\mathbb{E}[(2x_0' - 2x_0)^2] = 4 \cdot 2\operatorname{Var}(x_0)$이기 때문이다.
- `X`가 바뀌지 않고, 같은 seed면 같은 결과가 나온다.
- 출력과 상관된 특징이라도 **모델이 쓰지 않으면** 중요도가 0이다. permutation importance는 세계가 아니라 모델에 대한 설명이다.
