# Gram–Schmidt 직교화

열이 선형독립인 $m \times n$ 행렬 $\mathbf{A} = [\mathbf{a}_1, \dots, \mathbf{a}_n]$ ($n \le m$)에서, 같은 공간을 span하는 정규직교 열을 가진 행렬 $\mathbf{Q}$를 만든다. 각 열에서 앞서 만든 방향으로의 사영을 빼고 길이를 1로 맞춘다.

$$
\tilde{\mathbf{a}}_k = \mathbf{a}_k - \sum_{j < k} (\mathbf{q}_j^\top\mathbf{a}_k)\,\mathbf{q}_j, \qquad \mathbf{q}_k = \frac{\tilde{\mathbf{a}}_k}{\lVert\tilde{\mathbf{a}}_k\rVert}.
$$

함수 `gram_schmidt(A)`를 완성해 `(Q, R)`을 반환한다.

| 반환 | shape | 조건 |
|---|---|---|
| `Q` | `(m, n)` | $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ |
| `R` | `(n, n)` | 상삼각, 대각 성분 양수, $\mathbf{Q}\mathbf{R} = \mathbf{A}$ |

$\mathbf{R}$의 $(j, k)$ 성분은 $\mathbf{q}_j^\top\mathbf{a}_k$ ($j < k$), 대각 성분은 $\lVert\tilde{\mathbf{a}}_k\rVert$이다.

테스트는 다음을 확인한다.

- $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, $\mathbf{Q}\mathbf{R} = \mathbf{A}$, $\mathbf{R}$이 상삼각이고 대각이 양수
- 첫 열은 $\mathbf{a}_1$을 정규화한 것
- 열이 거의 평행한 행렬에서도 직교성이 $10^{-8}$ 수준으로 유지된다

힌트: 마지막 조건은 수식 그대로(classical) 구현하면 통과하지 못할 수 있다. 사영을 뺄 때 원래 $\mathbf{a}_k$ 대신, 지금까지 빼고 남은 벡터 $\tilde{\mathbf{a}}_k$에 대해 $\mathbf{q}_j^\top\tilde{\mathbf{a}}_k$를 계산해 빼는 방식(modified Gram–Schmidt)은 수학적으로 같은 결과를 주지만 반올림 오차에 더 강하다. 그래도 부족하면 같은 과정을 한 번 더 반복한다(재직교화).
