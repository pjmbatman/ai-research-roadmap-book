# scaled dot-product attention 구현하기

query $\mathbf{Q}$, key $\mathbf{K}$, value $\mathbf{V}$에 대한 scaled dot-product attention은

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)\mathbf{V}
$$

이다. softmax는 마지막 축(key 축)을 따라 계산한다. $D$는 query·key의 마지막 축 크기다.

함수 `attention(Q, K, V, mask=None)`을 완성해 `(output, weights)`를 반환한다.

| 인자·반환 | shape | 의미 |
|---|---|---|
| `Q` | `(..., T, D)` | query. 앞쪽 `...`은 batch·head 같은 임의의 축 |
| `K` | `(..., S, D)` | key |
| `V` | `(..., S, Dv)` | value |
| `mask` | `(T, S)` 또는 `(..., T, S)`, bool | `True`인 칸만 볼 수 있다. `None`이면 모두 본다 |
| `output` | `(..., T, Dv)` | 가중합 |
| `weights` | `(..., T, S)` | softmax 가중치. 행마다 합 1, 가려진 칸은 정확히 0 |

테스트는 다음을 확인한다.

- 반복문으로 정의를 그대로 계산한 값과 같다(batch·head 축이 있는 경우 포함).
- 가중치의 행 합이 1이고, `mask`가 `False`인 칸의 가중치가 정확히 0이다.
- 점수가 매우 커도(예: query·key 성분이 수백) `nan`이나 `inf` 없이 계산된다.

힌트: `np.swapaxes(K, -1, -2)`로 마지막 두 축만 바꾸면 앞쪽 축은 그대로 batch 축이 된다. softmax는 행마다 최댓값을 뺀 뒤 지수를 취한다. 가릴 칸은 점수를 `-np.inf`로 바꾸면 지수가 정확히 0이 된다.
