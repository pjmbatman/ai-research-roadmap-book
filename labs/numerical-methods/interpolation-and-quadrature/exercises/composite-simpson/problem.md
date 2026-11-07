# 합성 Simpson 규칙

구간 $[a, b]$를 폭 $h = (b-a)/n$인 작은 구간 $n$개로 나누고($n$은 짝수), 이웃한 두 구간마다 Simpson 규칙을 적용해 더하면

$$
\int_a^b f(x)\,dx \approx \frac{h}{3}\Big[f(x_0) + 4\sum_{i\ \text{홀수}} f(x_i) + 2\sum_{\substack{i\ \text{짝수} \\ 0<i<n}} f(x_i) + f(x_n)\Big], \qquad x_i = a + i h
$$

이다. 함수 `composite_simpson(f, a, b, n)`을 완성한다.

- `f`는 NumPy 배열을 받아 같은 shape의 배열을 돌려주는 함수다(예: `np.sin`).
- `n`이 짝수가 아니거나 2보다 작으면 `ValueError`를 낸다.
- 반환값은 실수(float) 하나다.

테스트는 다음을 확인한다.

1. 3차 이하 다항식은 반올림 오차 범위에서 정확히 적분한다(Simpson 규칙의 정확도 차수).
2. $\int_0^\pi \sin x\,dx = 2$에서 $n$을 두 배로 늘리면 오차가 약 16배 줄어든다(수렴 차수 4).
3. 홀수 $n$에는 `ValueError`를 낸다.

힌트: `x = np.linspace(a, b, n + 1)`로 노드를 만들고, 가중치 배열 `[1, 4, 2, 4, ..., 2, 4, 1]`을 만들어 `f(x)`와 내적한다.
