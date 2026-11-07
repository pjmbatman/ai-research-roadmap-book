# 비용을 최소화하는 threshold 고르기

분류기가 샘플마다 양성 점수 $p_i$를 낸다. threshold $t$를 정하면 $p_i \ge t$인 샘플을 양성으로 판정한다. 양성을 놓치는 비용(false negative)을 $c_\mathrm{FN}$, 음성을 양성으로 잘못 판정하는 비용(false positive)을 $c_\mathrm{FP}$라 하면 총비용은

$$
C(t) = c_\mathrm{FN}\cdot\#\{i : y_i = 1,\ p_i < t\} + c_\mathrm{FP}\cdot\#\{i : y_i = 0,\ p_i \ge t\}
$$

이다. 함수 `best_threshold(y, p, c_fn, c_fp)`를 완성해 총비용이 가장 작은 `(t, cost)`를 반환한다.

- 후보 threshold는 `p`에 나오는 서로 다른 값들과 `np.inf`(아무것도 양성으로 판정하지 않음)다. 이 후보들만 보면 충분하다. 그 사이의 $t$는 이 중 하나와 같은 판정을 내기 때문이다.
- 총비용이 같은 후보가 여럿이면 **가장 작은** $t$를 고른다.
- `cost`는 위 식의 총비용(평균이 아니라 합)이다.

테스트는 손으로 계산할 수 있는 작은 예, 모든 후보를 직접 평가한 결과와의 비교, 그리고 보정된(calibrated) 확률에서 고른 threshold가 이론값 $c_\mathrm{FP}/(c_\mathrm{FP} + c_\mathrm{FN})$ 근처인지를 확인한다.

힌트: `np.unique(p)`로 후보를 정렬된 순서로 얻고, 후보마다 비용을 계산해 `np.argmin`을 쓰면 된다(첫 번째 최솟값을 고른다). 샘플이 수천 개면 후보도 수천 개이므로, 양성·음성 점수를 각각 정렬해 두고 `np.searchsorted`로 "$t$보다 작은 점수의 개수"를 한 번에 세면 빠르다.
