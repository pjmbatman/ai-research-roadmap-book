# 예측 구간의 interval score

중심 $(1-\alpha)$ 예측 구간 $[l, u]$를 관측 $y$로 평가하는 interval score는

$$
S_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)\,\mathbb{1}[y < l] + \frac{2}{\alpha}(y - u)\,\mathbb{1}[y > u]
$$

이다(작을수록 좋다). 첫째 항은 구간의 폭(sharpness)이고, 나머지 두 항은 관측이 구간 밖에 떨어졌을 때 벗어난 거리에 $2/\alpha$를 곱한 벌점이다. 이 점수는 적정 점수 규칙이다. 기대 점수를 가장 작게 만드는 구간은 예측분포의 $\alpha/2$ 분위수와 $1-\alpha/2$ 분위수다.

두 함수를 완성한다.

1. `interval_score(lower, upper, y, alpha)`: 배열 `lower`, `upper`, `y`(같은 shape)에 대한 **평균** interval score
2. `best_symmetric_halfwidth(y, center, alpha, grid)`: 중심을 `center`로 고정하고 반폭 `h`만 바꿀 때, 후보 배열 `grid` 중 평균 interval score를 가장 작게 만드는 `h`

테스트는 손으로 계산한 예와, 표준정규 관측에서 2번 함수의 답이 $z_{1-\alpha/2}$ 근처인지 확인한다. 이 결과는 "구간을 일부러 좁히거나 넓혀서 점수를 올릴 수 없다"는 적정성의 의미다.
