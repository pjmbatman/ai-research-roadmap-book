# 상태공간 모델의 재귀와 합성곱

이산 선형 상태공간 모델

$$
\mathbf{x}_k = \bar{\mathbf{A}}\mathbf{x}_{k-1} + \bar{\mathbf{B}}u_k, \qquad y_k = \mathbf{C}\mathbf{x}_k, \qquad \mathbf{x}_{-1} = \mathbf{0}
$$

은 같은 출력을 두 방법으로 계산할 수 있다. 한 step씩 상태를 갱신하는 **재귀**(추론에 알맞음)와, kernel $K_j = \mathbf{C}\bar{\mathbf{A}}^j\bar{\mathbf{B}}$로 입력을 한 번에 거르는 **합성곱** $y_k = \sum_{j=0}^{k} K_j u_{k-j}$(병렬 학습에 알맞음)이다.

세 함수를 완성한다. `A_bar`의 shape은 `(N, N)`, `B_bar`는 `(N, 1)`, `C`는 `(1, N)`, `u`는 `(L,)`이다.

| 함수 | 반환 | shape |
|---|---|---|
| `ssm_scan(A_bar, B_bar, C, u)` | 재귀로 계산한 출력 $y_0, \dots, y_{L-1}$ | `(L,)` |
| `ssm_kernel(A_bar, B_bar, C, L)` | kernel $K_0, \dots, K_{L-1}$ | `(L,)` |
| `ssm_conv(A_bar, B_bar, C, u)` | `ssm_kernel`과 인과(causal) 합성곱으로 계산한 출력 | `(L,)` |

테스트는 다음을 확인한다.

- 무작위로 만든 안정한 모델에서 `ssm_scan`과 `ssm_conv`의 출력이 같다.
- `ssm_kernel`이 단위 충격 입력 $u = (1, 0, 0, \dots)$에 대한 `ssm_scan`의 출력과 같다(kernel은 impulse response다).
- 출력이 인과적이다. $k$ 이후의 입력을 바꿔도 $y_0, \dots, y_k$는 변하지 않는다.

힌트: `ssm_kernel`에서는 행렬 거듭제곱을 매번 새로 계산하지 말고 $\bar{\mathbf{A}}^j\bar{\mathbf{B}}$를 한 step씩 갱신한다. `ssm_conv`는 `np.convolve(u, K)[:L]`로 쓸 수 있다(왜 앞의 $L$개만 쓰는지 생각해 본다).
