# 혼합 가우시안의 평균 NLL을 안정하게 계산하기

생성모델을 likelihood로 평가하려면 test 데이터의 평균 음의 로그우도(NLL)

$$
\mathrm{NLL} = -\frac{1}{N}\sum_{i=1}^{N} \log p(\mathbf{x}_i), \qquad
p(\mathbf{x}) = \sum_{k=1}^{K} w_k\, \mathcal{N}\big(\mathbf{x};\ \boldsymbol{\mu}_k,\ \sigma^2 \mathbf{I}_d\big)
$$

를 계산해야 한다. 함수 `mixture_nll(x, centers, sd, weights=None)`를 완성한다.

| 인자 | shape·의미 |
|---|---|
| `x` | `(N, d)` 평가할 점 |
| `centers` | `(K, d)` 성분 평균 $\boldsymbol{\mu}_k$ |
| `sd` | 양수 스칼라, 모든 성분이 같은 등방 표준편차 $\sigma$ |
| `weights` | `(K,)` 성분 가중치. `None`이면 균등. 합이 1이 아니면 합으로 나눠 정규화한다 |

반환값은 nats 단위의 스칼라(float)다.

- $d$차원 등방 가우시안의 로그밀도는 $-\lVert\mathbf{x} - \boldsymbol{\mu}\rVert^2/(2\sigma^2) - \tfrac{d}{2}\log(2\pi\sigma^2)$이다.
- 점이 모든 성분에서 멀면 $\exp$가 0으로 underflow해 $\log 0 = -\infty$가 된다. 테스트에는 성분에서 표준편차의 수백 배 떨어진 점이 있다. 로그 공간에서 더하는 `logsumexp`(`scipy.special.logsumexp`)를 쓴다.
