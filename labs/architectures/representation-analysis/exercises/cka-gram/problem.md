# Gram 행렬로 CKA 계산하기

본문은 linear CKA를 특징 공간에서 계산했다. 같은 값을 표본 사이의 유사도 행렬(Gram 행렬)로도 쓸 수 있다. 같은 입력 $N$개에 대한 두 표현 $\mathbf{A} \in \mathbb{R}^{N \times D_1}$, $\mathbf{B} \in \mathbb{R}^{N \times D_2}$에서

$$
\mathbf{K} = \mathbf{A}\mathbf{A}^\top, \quad \mathbf{L} = \mathbf{B}\mathbf{B}^\top, \quad
\mathbf{C} = \mathbf{I}_N - \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top,
$$

$$
\mathrm{HSIC}(\mathbf{K}, \mathbf{L}) = \operatorname{tr}(\mathbf{C}\mathbf{K}\mathbf{C}\mathbf{L}), \qquad
\mathrm{CKA}(\mathbf{K}, \mathbf{L}) = \frac{\mathrm{HSIC}(\mathbf{K}, \mathbf{L})}{\sqrt{\mathrm{HSIC}(\mathbf{K}, \mathbf{K})\,\mathrm{HSIC}(\mathbf{L}, \mathbf{L})}}
$$

이다. $\mathbf{C}$는 중심화 행렬이다. (HSIC의 정규화 상수 $1/(N-1)^2$은 분자와 분모에서 약분되므로 생략했다.)

함수 `cka_gram(K, L)`를 완성한다. 인자는 두 Gram 행렬 `K`, `L`(shape `(N, N)`)이고 반환값은 float 하나다.

테스트는 다음을 확인한다.

- 선형 Gram 행렬($\mathbf{A}\mathbf{A}^\top$, $\mathbf{B}\mathbf{B}^\top$)에서 특징 공간 공식 $\lVert \mathbf{A}_c^\top\mathbf{B}_c \rVert_F^2 / (\lVert \mathbf{A}_c^\top\mathbf{A}_c \rVert_F \lVert \mathbf{B}_c^\top\mathbf{B}_c \rVert_F)$와 같다($\mathbf{A}_c$, $\mathbf{B}_c$는 열 중심화한 표현).
- 직교 변환과 등방 스케일에 불변이다.
- 자기 자신과의 CKA는 1이다.
- 차원이 다른 두 표현에도 쓸 수 있다(Gram 행렬은 둘 다 $N \times N$이다).

힌트: $\mathbf{C}\mathbf{K}\mathbf{C}$는 `K - K.mean(0) - K.mean(1)[:, None] + K.mean()`으로도 계산할 수 있고, $\operatorname{tr}(\mathbf{X}\mathbf{Y}) = \sum_{ij} X_{ij}Y_{ji}$다.
