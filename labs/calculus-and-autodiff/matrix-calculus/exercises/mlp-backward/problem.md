# 2층 MLP의 gradient를 손으로 구현하기

입력 batch $\mathbf{X} \in \mathbb{R}^{N \times D}$(행이 샘플)에 대한 2층 MLP와 손실은 다음과 같다.

$$
\mathbf{A} = \mathbf{X}\mathbf{W}_1^\top + \mathbf{1}\mathbf{b}_1^\top, \quad
\mathbf{H} = \mathrm{ReLU}(\mathbf{A}), \quad
\mathbf{Z} = \mathbf{H}\mathbf{W}_2^\top + \mathbf{1}\mathbf{b}_2^\top, \quad
L = -\frac{1}{N}\sum_{n} \log \operatorname{softmax}(\mathbf{z}_n)_{y_n}.
$$

가중치 shape은 PyTorch의 `nn.Linear`와 같은 (출력 × 입력)이다.

| 인자 | shape |
|---|---|
| `X` | `(N, D)` |
| `y` | `(N,)` 정수 레이블 |
| `W1`, `b1` | `(H, D)`, `(H,)` |
| `W2`, `b2` | `(K, H)`, `(K,)` |

함수 `mlp_grads(X, y, W1, b1, W2, b2)`를 완성해 `{"W1": ..., "b1": ..., "W2": ..., "b2": ...}` 딕셔너리를 반환한다. 각 gradient의 shape은 해당 파라미터와 같아야 한다.

본문의 유도를 따라 출력 쪽에서 입력 쪽으로 계산한다.

1. $\mathbf{G}_Z = (\mathbf{P} - \mathbf{Y})/N$ — softmax cross-entropy의 gradient
2. $\nabla \mathbf{W}_2 = \mathbf{G}_Z^\top \mathbf{H}$, $\nabla \mathbf{b}_2 = \mathbf{G}_Z^\top \mathbf{1}$
3. $\mathbf{G}_H = \mathbf{G}_Z \mathbf{W}_2$, $\mathbf{G}_A = \mathbf{G}_H \odot \mathbb{1}[\mathbf{A} > 0]$
4. $\nabla \mathbf{W}_1 = \mathbf{G}_A^\top \mathbf{X}$, $\nabla \mathbf{b}_1 = \mathbf{G}_A^\top \mathbf{1}$

테스트는 유한차분 gradient와 상대오차로 비교하고, shape과 몇 가지 성질(모든 은닉 뉴런이 꺼지면 첫 층의 gradient가 0)을 확인한다.

힌트: 각 단계에서 shape을 주석으로 적어 보면 전치를 어디에 붙일지 정해진다. $\mathbf{G}_Z^\top\mathbf{H}$는 $(K, N)(N, H) = (K, H)$다.
