# 합성곱을 행렬로 쓰기

길이 $T$인 신호 $\mathbf{x}$에 홀수 길이 $k$인 kernel $\mathbf{w}$를 원형(circular)으로 적용하는 1차원 합성곱을

$$
y_t = \sum_{j=0}^{k-1} w_j \, x_{(t + j - c) \bmod T}, \qquad c = \frac{k-1}{2}
$$

로 정의한다. kernel의 가운데 원소 $w_c$가 같은 위치 $x_t$에 곱해지고, 양쪽 끝은 신호의 반대쪽 끝으로 감긴다. 이 연산은 선형이므로 $\mathbf{y} = \mathbf{M}\mathbf{x}$인 $(T, T)$ 행렬 $\mathbf{M}$이 있다.

함수 `conv_matrix(kernel, T)`를 완성해 이 행렬 $\mathbf{M}$을 반환한다.

| 인자 | shape·타입 | 의미 |
|---|---|---|
| `kernel` | `(k,)`, $k$는 홀수, $k \le T$ | kernel $\mathbf{w}$ |
| `T` | `int` | 신호 길이 |

테스트는 다음을 확인한다.

- `conv_matrix(kernel, T) @ x`가 위 식을 그대로 계산한 값과 같다.
- 신호를 한 칸 민 뒤 합성곱한 결과와, 합성곱한 뒤 한 칸 민 결과가 같다(이동 등변성). 즉 $\mathbf{M}$이 이동 행렬 $\mathbf{S}$와 교환된다: $\mathbf{M}\mathbf{S} = \mathbf{S}\mathbf{M}$.
- 행렬의 각 행은 앞 행을 오른쪽으로 한 칸 원형 이동한 것이다(가중치 공유). 행렬에 서로 다른 값은 kernel의 값뿐이다.

힌트: 행 $t$에서 kernel 원소 $w_j$가 놓이는 열은 $(t + j - c) \bmod T$다. 반복문 두 개로 충분하다.
