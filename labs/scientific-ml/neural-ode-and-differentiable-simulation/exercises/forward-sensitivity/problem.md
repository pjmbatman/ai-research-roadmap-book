# solver를 통과하는 순방향 민감도

로지스틱 성장 모델

$$
y' = f(y; r) = r\,y\left(1 - \frac{y}{K}\right)
$$

을 explicit Euler $y_{n+1} = y_n + h\,f(y_n; r)$로 적분한다. 성장률 $r$에 대한 민감도 $s_n = \partial y_n / \partial r$는 Euler 식 자체를 $r$로 미분하면 얻는다.

$$
s_{n+1} = s_n + h\left(\frac{\partial f}{\partial y}(y_n; r)\,s_n + \frac{\partial f}{\partial r}(y_n; r)\right), \qquad s_0 = 0.
$$

이것은 "이산화한 다음 미분하기"(discretize-then-optimize)의 순방향 모드다. 계산한 $s_n$은 Euler로 만든 $y_n$의 **정확한** 도함수다.

함수 `euler_with_sensitivity(r, K, y0, h, steps)`를 완성해 `(y, s)`를 반환한다. 둘 다 shape `(steps + 1,)`이고 첫 원소는 초기값($y_0$, 0)이다.

테스트는 다음을 확인한다.

- $s_n$이 같은 Euler 시뮬레이션의 중심차분 $\big(y_n(r + \delta) - y_n(r - \delta)\big) / 2\delta$와 맞는다.
- $r = 0$이면 $y$는 변하지 않고 $s_n = n\,h\,y_0(1 - y_0/K)$다.
- 평형점 $y_0 = K$에서 출발하면 $s$는 계속 0이다.

힌트: $\partial f/\partial y = r(1 - 2y/K)$, $\partial f/\partial r = y(1 - y/K)$. $s_{n+1}$을 계산할 때는 갱신하기 **전**의 $y_n$을 쓴다.
