# 순열 중요도 구현하기

학습이 끝난 모델의 순열 중요도(permutation importance)는 평가 데이터에서 특징 하나(또는 여러 개)의 열을 무작위로 섞었을 때 손실이 얼마나 늘어나는지로 잰다.

$$
I_j = \frac{1}{R}\sum_{r=1}^{R} \Big[ L\big(f(\mathbf{X}^{(r)}_{\pi_j}), \mathbf{y}\big) \Big] - L\big(f(\mathbf{X}), \mathbf{y}\big).
$$

$\mathbf{X}^{(r)}_{\pi_j}$는 $r$번째 반복에서 열 $j$의 행 순서만 섞은 행렬이고, $L$은 평균 제곱오차다. 섞으면 특징과 목표의 관계가 끊기므로, 모델이 그 특징에 기대고 있을수록 손실이 크게 는다. 모델을 다시 학습하지 않는다는 점에 주의한다.

함수 `permutation_importance(predict, X, y, cols, n_repeats=10, seed=0)`를 완성한다.

| 인자 | 의미 |
|---|---|
| `predict` | `(N, D)` 배열을 받아 `(N,)` 예측을 반환하는 함수 |
| `X`, `y` | 평가 데이터 `(N, D)`, `(N,)` |
| `cols` | 섞을 열 번호의 리스트. 여러 개면 **같은 순열로 함께** 섞는다(상관된 특징 묶음) |
| `n_repeats`, `seed` | 반복 횟수와 `np.random.default_rng(seed)`의 시드 |

반환값은 MSE 증가량의 평균(실수 하나)이다. 원래 `X`를 바꾸지 않도록 복사본을 섞는다.

테스트는 독립 특징의 선형 모델 $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$에서 $I_j \approx 2w_j^2\operatorname{Var}(x_j)$인지, 모델이 쓰지 않는 특징의 중요도가 0인지, 여러 열을 함께 섞는지 확인한다.
