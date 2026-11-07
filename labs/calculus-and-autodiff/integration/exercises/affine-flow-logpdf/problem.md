# 아핀 flow의 로그밀도

가장 단순한 normalizing flow는 표준정규분포 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$에 가역 아핀 변환을 적용한 것이다.

$$
\mathbf{y} = \mathbf{A}\mathbf{z} + \mathbf{b}, \qquad \det\mathbf{A} \neq 0.
$$

밀도 변환 공식에 따라

$$
\log p_Y(\mathbf{y}) = \log p_Z\big(\mathbf{A}^{-1}(\mathbf{y} - \mathbf{b})\big) - \log\lvert\det\mathbf{A}\rvert,
\qquad
\log p_Z(\mathbf{z}) = -\tfrac{1}{2}\lVert\mathbf{z}\rVert^2 - \tfrac{d}{2}\log(2\pi)
$$

이다. 함수 `affine_flow_logpdf(Y, A, b)`를 완성한다.

| 인자 | shape |
|---|---|
| `Y` | `(N, d)`, 행마다 한 점 |
| `A` | `(d, d)`, 가역 |
| `b` | `(d,)` |

반환값은 shape `(N,)`인 로그밀도 배열이다.

테스트는 다음을 확인한다.

- 결과가 평균 $\mathbf{b}$, 공분산 $\mathbf{A}\mathbf{A}^\top$인 다변량 정규분포의 로그밀도와 같은가.
- $\det\mathbf{A} < 0$(반사가 섞인 변환)에서도 맞는가. 절댓값을 빠뜨리면 `log`에 음수가 들어간다.
- 2차원에서 밀도를 넓은 격자 위에서 적분하면 1이 되는가.

힌트: $\mathbf{A}^{-1}(\mathbf{y} - \mathbf{b})$는 역행렬을 만들지 말고 `np.linalg.solve(A, (Y - b).T).T`로 계산한다. $\log\lvert\det\mathbf{A}\rvert$는 `np.linalg.slogdet(A)`의 둘째 값이다.
