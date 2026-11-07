# bilinear layer를 einsum 한 줄로

두 입력 $\mathbf{x} \in \mathbb{R}^I$, $\mathbf{z} \in \mathbb{R}^J$을 받아 $K$개의 출력을 내는 bilinear layer는 출력마다 행렬 $\mathbf{W}_k \in \mathbb{R}^{I \times J}$ 하나를 써서

$$
y_k = \mathbf{x}^\top \mathbf{W}_k \mathbf{z} + c_k = \sum_{i,j} x_i \, W_{kij} \, z_j + c_k
$$

를 계산한다. 함수 `bilinear(x, W, z, c)`를 완성해 batch 전체의 출력을 반환한다.

| 인자 | shape | 의미 |
|---|---|---|
| `x` | `(B, I)` | 첫 번째 입력 |
| `W` | `(K, I, J)` | 출력마다 행렬 하나 |
| `z` | `(B, J)` | 두 번째 입력 |
| `c` | `(K,)` | bias |

반환값의 shape은 `(B, K)`다. 테스트는 이중 loop로 계산한 값과 비교하고, $I \neq J$인 경우와 $B = 1$인 경우도 확인한다.

힌트: 먼저 각 축에 글자를 붙인다(`b`, `i`, `j`, `k`). 출력에 남는 축과 합해서 사라지는 축을 나누면 einsum 문자열이 정해진다. bias는 broadcasting으로 더한다.
