# Dirichlet 에너지로 over-smoothing 진단하기

노드 표현이 이웃끼리 얼마나 다른지를 재는 **Dirichlet 에너지**는

$$
E(\mathbf{H}) = \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} A_{ij}\,\lVert \mathbf{h}_i - \mathbf{h}_j \rVert^2
$$

이다. $\mathbf{A}$는 대칭 인접행렬, $\mathbf{h}_i$는 $\mathbf{H}$의 $i$번째 행이다. 무방향 간선 하나는 $(i, j)$와 $(j, i)$로 두 번 더해지므로 $\tfrac{1}{2}$을 곱해, 간선마다 한 번씩 센 값과 같게 한다. 층을 지날수록 이 값이 0으로 떨어지면 over-smoothing이다.

함수 `dirichlet_energy(A, H)`를 완성한다.

| 인자 | shape | 설명 |
|---|---|---|
| `A` | `(N, N)` | 대칭 인접행렬(가중치가 있어도 된다) |
| `H` | `(N, D)` | 노드 표현 |

반환값은 float 하나다.

테스트는 다음을 확인한다.

- 모든 노드의 표현이 같으면 0이다.
- 간선 하나짜리 그래프에서 $\lVert \mathbf{h}_0 - \mathbf{h}_1 \rVert^2$과 같다.
- 노드 번호를 바꿔도($\mathbf{P}\mathbf{A}\mathbf{P}^\top$, $\mathbf{P}\mathbf{H}$) 값이 같다(순열 불변).
- 그래프 라플라시안 $\mathbf{L} = \mathbf{D} - \mathbf{A}$로 쓴 $\operatorname{tr}(\mathbf{H}^\top\mathbf{L}\mathbf{H})$와 같다.
- 정규화한 이웃 평균을 반복하면 값이 줄어든다.

힌트: 이중 반복문으로 써도 되고, `H[:, None] - H[None]`로 모든 쌍의 차이 `(N, N, D)`를 한 번에 만들어도 된다.
