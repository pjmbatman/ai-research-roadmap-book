# affine coupling flow의 log-likelihood

2차원 affine coupling 층 하나로 만든 flow를 생각한다. 기저 분포는 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$이고, 정변환 $\mathbf{x} = f(\mathbf{z})$는

$$
x_1 = z_1, \qquad x_2 = z_2 \exp\big(s(z_1)\big) + t(z_1)
$$

이다. $s$와 $t$는 임의의 함수(실제로는 신경망)이고, 배열을 받아 같은 shape의 배열을 돌려준다.

함수 `flow_log_prob(x, s_fn, t_fn)`을 완성해 각 점의 $\log p(\mathbf{x})$를 반환한다.

| 인자 | shape·의미 |
|---|---|
| `x` | `(N, 2)` 평가할 점 |
| `s_fn`, `t_fn` | `(N,)` 배열을 받아 `(N,)` 배열을 돌려주는 함수 |

반환값은 `(N,)` 배열이다.

순서대로 하면 된다.

1. **역변환**: $z_1 = x_1$, $z_2 = \big(x_2 - t(x_1)\big)\exp\big(-s(x_1)\big)$. $s$와 $t$의 역함수는 필요 없다.
2. **log-det**: 정변환의 Jacobian은 하삼각행렬 $\begin{bmatrix} 1 & 0 \\ \ast & e^{s(z_1)} \end{bmatrix}$이므로 $\log\lvert\det \partial\mathbf{x}/\partial\mathbf{z}\rvert = s(z_1)$이다.
3. **변수변환 공식**(8.3장): $\log p(\mathbf{x}) = \log \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I}) - \log\lvert\det \partial\mathbf{x}/\partial\mathbf{z}\rvert$.

부호에 주의한다. 정변환이 공간을 늘리는 곳($s > 0$)에서는 밀도가 **낮아져야** 한다. 테스트는 $s, t$가 상수일 때의 닫힌 꼴 가우시안, 격자 적분한 전체 질량, 표본의 구간 비율로 확인한다.
