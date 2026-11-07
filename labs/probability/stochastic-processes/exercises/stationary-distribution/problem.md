# 전이행렬의 정상분포 구하기

상태가 $K$개인 Markov chain의 전이행렬 $\mathbf{P}$는 $P_{ij} = \Pr(X_{t+1} = j \mid X_t = i)$이고, 각 행의 합이 1이다. 정상분포 $\boldsymbol{\pi}$는 **행벡터**로서

$$
\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1, \qquad \pi_i \ge 0
$$

을 만족한다. 함수 `stationary_distribution(P)`를 완성해 shape `(K,)`인 $\boldsymbol{\pi}$를 반환한다. 체인은 기약(irreducible)이라고 가정해도 된다. 즉 정상분포는 하나뿐이다.

테스트는 두 상태 체인의 공식 $\boldsymbol{\pi} = \big(b, a\big)/(a + b)$(여기서 $a = P_{12}$, $b = P_{21}$), 각 열의 합도 1인 이중 확률행렬(정상분포가 균등), 그리고 무작위 전이행렬에서 $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$를 확인한다.

힌트: $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$를 전치하면 $\mathbf{P}^\top \boldsymbol{\pi}^\top = \boldsymbol{\pi}^\top$, 즉 $\mathbf{P}^\top$의 고윳값 1에 대한 고유벡터다. $\mathbf{P}$의 고유벡터를 쓰면 모든 성분이 같은 벡터가 나온다(왜 그런지 생각해 본다). 고유벡터 대신 $(\mathbf{P}^\top - \mathbf{I})\boldsymbol{\pi}^\top = \mathbf{0}$에 $\sum_i \pi_i = 1$을 한 줄 더 붙여 최소제곱으로 풀어도 된다.
