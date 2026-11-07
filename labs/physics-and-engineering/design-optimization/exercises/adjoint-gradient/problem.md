# 열전도 설계의 adjoint 기울기

길이가 같은 요소 $n$개로 나눈 1차원 막대의 정상 열전도를 1차 유한요소로 푼다. 요소 $e$의 열전도도 $k_e$가 설계 변수다. 왼쪽 끝 절점의 온도는 0으로 고정하고, 나머지 절점 $1, \dots, n$의 온도 $\mathbf{T} \in \mathbb{R}^n$은

$$
\mathbf{K}(\mathbf{k})\,\mathbf{T} = \mathbf{f}
$$

를 푼다. $\mathbf{K}$는 요소 행렬 $\frac{k_e}{h}\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}$를 절점 $(e, e+1)$ 자리에 더해 만든 행렬에서 고정 절점 0의 행과 열을 뺀 $n \times n$ 행렬이다($h$는 요소 길이). 목적함수는 절점 온도의 가중합 $J = \mathbf{c}^\top\mathbf{T}$다.

함수 `adjoint_gradient(k, f, c, h)`를 완성해 `(J, grad)`를 반환한다. `grad[e]`는 $\partial J/\partial k_e$다.

adjoint 방법은 다음과 같다.

1. $\mathbf{K}\mathbf{T} = \mathbf{f}$를 푼다.
2. $\mathbf{K}^\top\boldsymbol{\lambda} = \mathbf{c}$를 푼다(adjoint 해석).
3. $\dfrac{\partial J}{\partial k_e} = -\boldsymbol{\lambda}^\top \dfrac{\partial \mathbf{K}}{\partial k_e}\mathbf{T}$. 고정 절점의 값을 0으로 붙인 $\tilde{\mathbf{T}} = (0, \mathbf{T})$, $\tilde{\boldsymbol{\lambda}} = (0, \boldsymbol{\lambda})$를 쓰면 $-\dfrac{1}{h}(\tilde\lambda_{e+1} - \tilde\lambda_e)(\tilde T_{e+1} - \tilde T_e)$이다.

테스트는 다음을 확인한다.

- 기울기가 유한차분과 맞는다(여러 $\mathbf{c}$에 대해).
- $\mathbf{c} = \mathbf{f}$(열 compliance)이면 $\boldsymbol{\lambda} = \mathbf{T}$이므로 모든 성분이 0 이하다. 어느 요소의 전도도를 올려도 compliance가 줄어든다.
- 설계 변수가 200개여도 선형계 풀이(`np.linalg.solve`) 호출이 3회 이하다. 유한차분은 201회가 필요하다.

힌트: $\mathbf{K}$의 대각 원소는 $(k_e + k_{e+1})/h$(마지막 절점은 $k_n/h$), 위·아래 대각 원소는 $-k_{e+1}/h$다(절점 번호 기준으로 맞춰 본다). `np.linalg.solve`를 두 번만 쓴다.
