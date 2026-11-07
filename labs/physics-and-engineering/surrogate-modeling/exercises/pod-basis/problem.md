# POD 기저를 에너지 기준으로 고르기

스냅샷 행렬 $\mathbf{X}$(행 하나가 시뮬레이션 결과 한 개, shape `(M, N)`)에서 평균 $\bar{\mathbf{x}}$를 빼고 특이값 분해하면

$$
\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top = \mathbf{U}\,\mathrm{diag}(s_1, \dots, s_K)\,\mathbf{V}^\top
$$

이다. $\mathbf{V}$의 열(코드에서는 `Vt`의 행)이 POD 모드이고, $s_k^2$이 모드 $k$가 담는 "에너지"(분산)다.

함수 `pod_basis(X, energy)`를 완성해 `(mean, modes)`를 반환한다.

- `mean`: shape `(N,)`, 스냅샷의 평균
- `modes`: shape `(r, N)`, 행이 정규직교인 POD 모드. 누적 에너지 비율 $\sum_{k \le r} s_k^2 / \sum_k s_k^2$이 `energy` 이상이 되는 **가장 작은** $r$을 고른다.

테스트는 다음을 확인한다.

- 모드가 정규직교다: `modes @ modes.T`가 단위행렬이다.
- 알려진 특이값을 가진 데이터에서 $r$을 올바르게 고른다.
- 스냅샷을 모드로 투영해 복원한 오차의 제곱합이 버린 에너지 $\sum_{k > r} s_k^2$과 같다.
- 모든 스냅샷이 같은 질량(성분 합)을 가지면 모드의 성분 합은 0이다. 그래서 평균 + 모드의 선형 결합으로 만든 예측은 질량을 정확히 보존한다.

힌트: `np.linalg.svd(Xc, full_matrices=False)`를 쓰고, 누적 에너지 비율에 `np.searchsorted`를 쓰거나 반복문으로 $r$을 찾는다. 반올림 오차 때문에 누적 비율이 `energy`와 거의 같을 때 판정이 흔들리지 않도록 아주 작은 허용 오차(예: `1e-12`)를 둔다.
