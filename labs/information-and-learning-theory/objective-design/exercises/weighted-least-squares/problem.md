# 우도에서 가중 최소제곱 유도하기

선형 모델 $y_i = \mathbf{x}_i^\top\boldsymbol\beta + \varepsilon_i$에서 측정마다 잡음의 표준편차 $\sigma_i$가 알려져 있다고 하자: $\varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)$, 서로 독립.

두 함수를 완성한다.

1. `gaussian_nll(beta, X, y, sigma)`: 이 모형의 평균 음의 로그우도 [nat]
   $$
   \frac{1}{N}\sum_{i=1}^{N}\left[\frac{(y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2}{2\sigma_i^2} + \log\sigma_i + \frac12\log 2\pi\right]
   $$
2. `weighted_lstsq(X, y, sigma)`: 위 NLL을 최소화하는 $\boldsymbol\beta$. $\mathbf{W} = \mathrm{diag}(1/\sigma_i^2)$로 두면 정규방정식은 $\mathbf{X}^\top\mathbf{W}\mathbf{X}\boldsymbol\beta = \mathbf{X}^\top\mathbf{W}\mathbf{y}$다.

| 인자 | shape |
|---|---|
| `X` | `(N, p)` |
| `y` | `(N,)` |
| `sigma` | `(N,)`, 모두 양수 |
| `beta` | `(p,)` |

테스트는 다음을 확인한다.

- `weighted_lstsq`의 해에서 NLL의 gradient가 0이다(다른 $\boldsymbol\beta$보다 NLL이 작다).
- 모든 $\sigma_i$가 같으면 보통의 최소제곱 해(`np.linalg.lstsq`)와 같다.
- $\sigma$ 전체에 상수를 곱해도 해는 변하지 않는다(가중치의 상대적 크기만 중요하다).
- 잡음이 아주 큰 측정 하나를 추가해도 해가 거의 변하지 않는다.

힌트: $\mathbf{W}$를 행렬로 만들지 않고 `X / sigma[:, None]`, `y / sigma`로 각 행을 나눈 뒤 보통의 최소제곱을 풀어도 같다.
