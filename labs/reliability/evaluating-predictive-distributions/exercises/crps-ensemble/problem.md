# 앙상블 예측의 CRPS

앙상블·MC dropout·생성모델처럼 예측분포를 **표본**으로만 내는 모델은 밀도를 모르므로 NLL을 바로 계산할 수 없다. CRPS는 표본만으로 계산할 수 있다. 예측분포 $F$에서 독립으로 뽑은 $X, X'$에 대해

$$
\mathrm{CRPS}(F, y) = \mathbb{E}\,\lvert X - y\rvert - \tfrac{1}{2}\,\mathbb{E}\,\lvert X - X'\rvert
$$

이다. 표본 $x_1, \dots, x_S$가 있으면 두 기대값을 표본 평균으로 바꾼다.

$$
\widehat{\mathrm{CRPS}} = \frac{1}{S}\sum_{i} \lvert x_i - y\rvert - \frac{1}{2S^2}\sum_{i}\sum_{j} \lvert x_i - x_j\rvert.
$$

함수 `crps_ensemble(samples, y)`를 완성한다.

| 인자 | shape | 의미 |
|---|---|---|
| `samples` | `(S, N)` | 관측 $N$개 각각에 대한 예측 표본 $S$개 |
| `y` | `(N,)` | 관측값 |

반환값은 관측마다의 CRPS, shape `(N,)`이다. 이중합은 `samples[:, None, :] - samples[None, :, :]`로 한 번에 계산할 수 있다(메모리 $S^2 N$).

테스트는 다음을 확인한다.

- 표본이 하나뿐이면 CRPS는 절댓값 오차 $\lvert x_1 - y\rvert$다.
- 가우시안 표본이 많으면 닫힌 형태(12.3장)에 가깝다.
- 관측값을 정확히 맞힌 퇴화 분포(모든 표본이 $y$)의 CRPS는 0이다.
