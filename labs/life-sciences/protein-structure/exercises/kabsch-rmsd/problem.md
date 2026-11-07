# Kabsch 정렬 RMSD

두 구조 $\mathbf{P}, \mathbf{Q} \in \mathbb{R}^{L \times 3}$(같은 행끼리 같은 잔기, 단위 Å)를 회전과 평행이동으로 가장 잘 겹친 뒤의 RMSD를 계산한다.

$$
\mathrm{RMSD}(\mathbf{P}, \mathbf{Q}) = \min_{\mathbf{R} \in SO(3),\, \mathbf{t}} \sqrt{\frac{1}{L} \sum_{i=1}^{L} \lVert \mathbf{R}^\top \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \rVert^2 }
$$

함수 `kabsch_rmsd(P, Q)`를 완성해 RMSD(실수 하나)를 반환한다.

1. 두 구조에서 각자의 무게중심(행 평균)을 뺀다: $\mathbf{P}_0, \mathbf{Q}_0$.
2. $\mathbf{H} = \mathbf{P}_0^\top \mathbf{Q}_0$ ($3 \times 3$)의 SVD $\mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$를 구한다.
3. $d = \operatorname{sign}\det(\mathbf{U}\mathbf{V}^\top)$로 두고 $\mathbf{R} = \mathbf{U}\,\mathrm{diag}(1, 1, d)\,\mathbf{V}^\top$. **반사는 허용하지 않는다**(단백질의 거울상은 다른 분자다).
4. $\mathbf{P}_0 \mathbf{R}$과 $\mathbf{Q}_0$의 RMSD를 반환한다.

테스트는 회전·평행이동한 구조의 RMSD가 0인지, 거울상이 0이 아닌지, 잡음 크기에 맞는 값이 나오는지, 무작위 회전을 여러 개 시도한 어떤 결과보다도 작거나 같은지 확인한다.

힌트: `np.linalg.svd`는 `U, S, Vt`를 반환한다. 세 번째 값이 이미 $\mathbf{V}^\top$이다.
