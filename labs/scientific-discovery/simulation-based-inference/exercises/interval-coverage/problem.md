# 사후 구간의 coverage 재기

사전분포에서 참값 $\tilde\theta_m$ ($m = 1, \dots, M$)을 뽑아 데이터를 시뮬레이션하고, 각 데이터로 추론해 사후 표본 $L$개를 얻었다. 사후 표본은 `samples` (shape `(M, L)`), 참값은 `theta_true` (shape `(M,)`)다.

함수 `coverage(samples, theta_true, level)`를 완성한다.

1. 각 행의 사후 표본으로 **중앙** `level` 구간 $[Q_{(1-\text{level})/2},\ Q_{(1+\text{level})/2}]$을 만든다. 분위수는 `np.quantile`의 기본 방식으로 계산한다.
2. 참값이 구간 안(양 끝 포함)에 있는 비율을 반환한다.

테스트는 다음을 확인한다.

- 정확한 사후분포에서 뽑은 표본이면 coverage가 `level`에 가까운지(50%, 90%)
- 사후분포를 일부러 너무 좁게 만들면(과신) coverage가 크게 떨어지는지
- 사후분포를 너무 넓게 만들면(보수적) coverage가 `level`보다 높아지는지

테스트에 쓰는 모델은 정규-정규 켤레 모델이다. $\theta \sim \mathcal{N}(0, 1)$, $x \mid \theta \sim \mathcal{N}(\theta, 1)$이면 정확한 사후분포는 $\mathcal{N}(x/2, 1/2)$다(13.2장).
