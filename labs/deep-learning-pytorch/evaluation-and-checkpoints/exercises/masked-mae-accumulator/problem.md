# mask가 있는 MAE를 batch 단위로 집계하기

길이가 다른 시계열을 padding해 batch로 묶으면, 각 batch는 예측 `pred`, 정답 `target`, 관측 여부 `mask`를 가진다(shape은 모두 `(B, T)`, `mask`는 1이 관측·0이 padding). batch마다 `B`와 `T`가 다를 수 있다.

함수 `masked_mae(batches)`를 완성한다. `batches`는 `(pred, target, mask)` 튜플의 리스트이고, 반환값은 **관측된 모든 값에 대한** 평균 절대 오차다.

$$
\mathrm{MAE} = \frac{\sum_{b}\sum_{i,t} m^{(b)}_{it}\,\lvert \hat{y}^{(b)}_{it} - y^{(b)}_{it}\rvert}{\sum_{b}\sum_{i,t} m^{(b)}_{it}}
$$

- batch 평균을 다시 평균하지 않는다. 분자와 분모를 따로 누적한다.
- padding 위치의 값은 무엇이든(0, 큰 수, NaN) 결과에 영향을 주면 안 된다.
- `mask`가 `None`이면 그 batch의 모든 값이 관측이다.
- 관측값이 하나도 없으면 `ValueError`를 낸다(0으로 나누어 `nan`을 조용히 반환하지 않는다).

테스트는 batch를 하나로 이어 붙여 계산한 값과 비교하고, batch 크기가 고르지 않은 경우와 padding에 NaN이 들어 있는 경우를 확인한다.

힌트: `np.where(mask > 0, np.abs(pred - target), 0.0)`은 padding 위치의 NaN도 0으로 바꾼다. `mask * np.abs(...)`는 `0 * nan = nan`이라 그렇지 않다.
