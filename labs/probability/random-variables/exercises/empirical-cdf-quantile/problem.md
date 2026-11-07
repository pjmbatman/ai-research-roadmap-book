# 경험적 CDF와 quantile function

표본 $x_1, \dots, x_n$의 **경험적 CDF**는

$$
F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[x_i \le x]
$$

이다. 계단함수이므로 보통의 역함수가 없다. 대신 일반화 역함수

$$
Q_n(p) = \min\{\, x_i : F_n(x_i) \ge p \,\}, \qquad 0 < p \le 1
$$

를 quantile function으로 쓴다. "누적 비율이 처음으로 $p$에 닿는 표본값"이다. 두 함수를 완성한다.

1. `ecdf(samples, x)`: 표본 배열 `samples` (shape `(n,)`)와 평가할 점 `x`(스칼라 또는 배열)에 대해 $F_n(x)$를 `x`와 같은 shape으로 반환한다.
2. `quantile(samples, p)`: `p`(스칼라 또는 배열, 각 값은 $0 < p \le 1$)에 대해 $Q_n(p)$를 `p`와 같은 shape으로 반환한다.

테스트는 다음을 확인한다.

- 같은 값이 여러 번 나오는 표본(동점)에서도 $F_n$이 맞는다.
- $F_n(Q_n(p)) \ge p$, 그리고 표본값 $x$에 대해 $Q_n(F_n(x)) \le x$. CDF와 quantile function을 오갈 때 성립하는 관계다.
- 큰 표본에서 $F_n$이 참 CDF에 가깝다.

힌트: 정렬한 표본 `s`에서 $F_n(x)$는 "`x` 이하인 원소의 개수 / n"이고, `np.searchsorted(s, x, side="right")`가 그 개수를 준다. $Q_n(p)$는 정렬한 표본의 $\lceil np \rceil$번째 값(1부터 셀 때)이다. 부동소수점 오차로 `n * p`가 정수보다 아주 조금 커지는 경우에 주의한다.
