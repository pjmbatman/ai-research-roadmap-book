# leverage와 Cook's distance로 영향점 찾기

선형회귀에서 관측 하나가 적합 결과를 얼마나 바꾸는지 재는 두 함수를 완성한다.

1. `leverage(X)`: hat 행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$의 대각 성분 $h_{ii}$, shape `(n,)`.
   $h_{ii}$는 관측 $i$의 $y_i$가 자기 적합값 $\hat{y}_i$에 주는 가중치다. 입력 $\mathbf{x}_i$가 다른 점들에서 멀수록 크다.
2. `cooks_distance(X, y)`: 각 관측의 Cook's distance, shape `(n,)`.
   $$
   D_i = \frac{e_i^2}{p\,\hat\sigma^2}\cdot\frac{h_{ii}}{(1 - h_{ii})^2}
   $$
   $e_i$는 최소제곱 잔차, $p$는 계수의 수, $\hat\sigma^2 = \sum_i e_i^2 / (n - p)$다.

Cook's distance는 "관측 $i$를 빼고 다시 적합하면 적합값 전체가 얼마나 움직이는가"를 한 숫자로 요약한 것이다. 위 식은 실제로 $n$번 다시 적합하지 않고 같은 값을 주는 닫힌 꼴이다. 테스트가 이 사실을 확인한다.

$$
D_i = \frac{\lVert \hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)} \rVert^2}{p\,\hat\sigma^2}
$$

여기서 $\hat{\mathbf{y}}_{(i)}$는 관측 $i$를 빼고 적합한 계수로 모든 점에서 계산한 적합값이다.
