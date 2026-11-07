# 보고된 단일 실행값이 재현 범위 안에 있는가

논문은 한 번 실행한 값 하나를 보고했고, 나는 같은 조건에서 seed $n$개로 재현해 값 $x_1, \dots, x_n$을 얻었다. 평균 $\bar{x}$, 표본 표준편차 $s$(`ddof=1`), $t = t_{(1+\text{level})/2,\; n-1}$이라 할 때 세 함수를 완성한다.

1. `mean_interval(runs, level=0.95)`: **평균**의 신뢰구간 $\bar{x} \pm t\, s/\sqrt{n}$을 `(lo, hi)`로 반환한다.
2. `single_run_interval(runs, level=0.95)`: **새로 한 번 실행한 값**의 예측구간 $\bar{x} \pm t\, s\sqrt{1 + 1/n}$을 `(lo, hi)`로 반환한다.
3. `is_consistent(reported, runs, level=0.95)`: 보고값이 예측구간 안에 있으면 `True`.

$t$ 분위수는 `scipy.stats.t.ppf`로 구한다. 테스트는 시뮬레이션으로, 예측구간은 새 단일 실행값을 약 95% 덮지만 평균의 신뢰구간은 그보다 훨씬 적게 덮는다는 것을 확인한다. 단일 실행값을 평균의 신뢰구간과 비교하면, 제대로 재현된 결과도 "재현 실패"로 판정하게 된다.
