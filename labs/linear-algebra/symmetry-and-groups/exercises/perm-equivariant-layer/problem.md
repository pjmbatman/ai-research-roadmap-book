# 순열 등변 층 만들고 검증하기

집합(점구름, 분자의 원자들, 입자들)은 원소의 순서에 의미가 없다. 원소 $N$개의 특징을 행으로 쌓은 행렬 $\mathbf{H} \in \mathbb{R}^{N \times D_{\text{in}}}$에 대해, 순서를 바꾸는 치환 행렬 $\mathbf{P}$를 적용하면 출력의 행도 똑같이 바뀌어야 한다.

$$
f(\mathbf{P}\mathbf{H}) = \mathbf{P}\,f(\mathbf{H}).
$$

다음 층은 이 성질(순열 등변성)을 만족한다.

$$
f(\mathbf{H}) = \mathbf{H}\mathbf{W}_{\text{self}} + \mathbf{1}_N\,\bar{\mathbf{h}}^\top\mathbf{W}_{\text{pool}} + \mathbf{1}_N\mathbf{b}^\top,
\qquad \bar{\mathbf{h}} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{h}_n.
$$

첫 항은 원소마다 같은 가중치를 따로 적용하고, 둘째 항은 집합 전체의 평균을 모든 원소에 똑같이 더한다.

함수 `equivariant_layer(H, W_self, W_pool, b)`를 완성한다. 입력에는 batch 축이 있다.

| 인자 | shape |
|---|---|
| `H` | `(B, N, D_in)` |
| `W_self` | `(D_in, D_out)` |
| `W_pool` | `(D_in, D_out)` |
| `b` | `(D_out,)` |

반환값의 shape은 `(B, N, D_out)`이다.

테스트는 다음을 확인한다.

- 위 식과 같은 값을 낸다.
- batch 항목마다 원소 순서를 무작위로 바꿔도 출력이 같은 방식으로 바뀐다(등변).
- 평균은 **batch 항목마다, 원소 축으로** 낸다. batch 축이나 feature 축으로 평균을 내면 등변성 또는 batch 독립성이 깨진다.
- 같은 가중치로 원소 수가 다른 집합($N = 3$, $N = 10$)을 처리한다.
- 출력을 원소 축으로 합하면 순서에 불변인 집합 표현이 된다.

힌트: `H.mean(axis=?, keepdims=True)`의 shape이 `(B, 1, D_in)`이 되도록 축을 고른다. 행렬 곱 `@`은 마지막 두 축에 작용하고 앞 축은 broadcasting된다(7.7장).
