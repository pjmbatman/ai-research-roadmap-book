# chain rule로 2층 신경망의 Jacobian 유도하기

입력 $\mathbf{x} \in \mathbb{R}^n$을 받는 2층 신경망

$$
g(\mathbf{x}) = \mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

를 세 사상의 합성 $\mathbf{x} \mapsto \mathbf{a} = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 \mapsto \mathbf{h} = \tanh(\mathbf{a}) \mapsto \mathbf{W}_2\mathbf{h} + \mathbf{b}_2$로 본다. 각 사상의 Jacobian을 구하고 chain rule로 곱해 $\mathbf{J}_g(\mathbf{x})$를 계산하는 함수 `mlp_jacobian(W1, b1, W2, b2, x)`를 완성한다.

| 인자 | shape |
|---|---|
| `W1` | `(k, n)` |
| `b1` | `(k,)` |
| `W2` | `(m, k)` |
| `b2` | `(m,)` |
| `x` | `(n,)` |

반환값은 shape `(m, n)`이다. 유한차분을 쓰지 말고 해석적으로 계산한다. 테스트는 유한차분 Jacobian과 비교하고, shape과 몇 가지 성질을 확인한다.

힌트: $\tanh'(a) = 1 - \tanh(a)^2$이다. 성분별 함수의 Jacobian은 대각행렬이다. 대각행렬을 만들지 않고 `(1 - h**2)[:, None] * W1`처럼 행마다 곱해도 같다.
