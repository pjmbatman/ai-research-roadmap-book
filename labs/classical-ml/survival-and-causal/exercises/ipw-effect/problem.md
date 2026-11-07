# 성향점수 가중으로 처치 효과 추정하기

관찰 데이터에서 처치 $T \in \{0, 1\}$(예: 첨가제 사용)의 평균 효과 $E[Y \mid do(T=1)] - E[Y \mid do(T=0)]$를 추정한다. 교란 변수 $Z$가 처치와 결과에 함께 영향을 주므로 단순 비교는 편향된다. $Z$가 backdoor 경로를 모두 막는다고 가정하면, 성향점수 $e(Z) = P(T = 1 \mid Z)$로 가중해 이 편향을 없앨 수 있다. 세 함수를 완성한다.

1. `naive_difference(y, t)`: 처치군 평균 − 대조군 평균.
2. `propensity(Z, t, iters=25)`: 절편을 포함한 로지스틱 회귀 $e = \sigma(w_0 + \mathbf{w}^\top\mathbf{z})$를 Newton 방법으로 적합해 샘플마다의 $e_i$, shape `(N,)`을 반환한다. `Z`는 shape `(N, p)`다. Newton 갱신은
   $$\mathbf{w} \leftarrow \mathbf{w} + (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{t} - \mathbf{e}), \qquad \mathbf{W} = \operatorname{diag}(e_i(1 - e_i))$$
   이고 $\mathbf{X}$는 `Z` 앞에 1의 열을 붙인 행렬, 초기값은 $\mathbf{w} = \mathbf{0}$이다.
3. `ipw_effect(y, t, e)`: 정규화한 역확률 가중(IPW) 추정량
   $$\hat{\tau} = \frac{\sum_i t_i y_i / e_i}{\sum_i t_i / e_i} - \frac{\sum_i (1 - t_i) y_i / (1 - e_i)}{\sum_i (1 - t_i) / (1 - e_i)}.$$

테스트는 손으로 계산할 수 있는 예, 로지스틱 회귀의 기울기 조건($\mathbf{X}^\top(\mathbf{t} - \mathbf{e}) = \mathbf{0}$), 그리고 교란이 있는 시뮬레이션에서 단순 비교는 크게 빗나가고 IPW는 참 효과에 가까운지를 확인한다.
