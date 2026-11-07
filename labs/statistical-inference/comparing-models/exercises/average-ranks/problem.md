# 여러 데이터셋에서의 평균 순위와 critical difference

방법 $k$개를 데이터셋 $N$개에서 비교할 때는 점수를 평균하기보다, 데이터셋마다 순위를 매긴 뒤 순위를 평균한다(Demšar, 2006). 세 함수를 완성한다.

1. `average_ranks(scores, lower_is_better=True)`
   - `scores`는 shape `(N, k)` 배열이다. `scores[i, j]`는 데이터셋 $i$에서 방법 $j$의 점수다.
   - 데이터셋(행)마다 가장 좋은 방법이 순위 1이다. `lower_is_better=True`이면 작은 점수가 좋다(오차), `False`이면 큰 점수가 좋다(정확도).
   - **동점**이면 그 순위들의 평균을 준다. 예: 두 방법이 공동 1위면 둘 다 1.5.
   - 방법마다 평균 순위, 즉 shape `(k,)` 배열을 반환한다.
2. `friedman_chi2(avg_ranks, n_datasets)`: Friedman 통계량
   $$\chi^2_F = \frac{12N}{k(k+1)}\left[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\right]$$
   $R_j$는 방법 $j$의 평균 순위다.
3. `critical_difference(k, n_datasets, q_alpha)`: Nemenyi 사후 검정의 critical difference
   $$\mathrm{CD} = q_\alpha\sqrt{\frac{k(k+1)}{6N}}$$
   두 방법의 평균 순위 차이가 CD보다 크면 차이가 유의하다고 본다.
