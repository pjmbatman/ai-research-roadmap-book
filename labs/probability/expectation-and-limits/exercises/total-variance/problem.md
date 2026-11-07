# 전체 분산을 배치 안과 배치 간으로 나누기

측정을 $G$개의 배치(장비·날짜·시료 묶음)로 나눠 배치마다 $n$번씩 했다. 측정값 배열 `y`의 shape은 `(G, n)`이다. 전체 분산의 법칙

$$
\operatorname{Var}(Y) = \mathbb{E}\big[\operatorname{Var}(Y \mid \text{배치})\big] + \operatorname{Var}\big(\mathbb{E}[Y \mid \text{배치}]\big)
$$

을 표본에 적용해, 함수 `variance_decomposition(y)`가 `(total, within, between)`을 반환하게 한다.

| 반환값 | 의미 |
|---|---|
| `total` | 모든 측정값을 하나로 모은 분산 |
| `within` | 배치마다 구한 분산의 평균 (배치 안 분산) |
| `between` | 배치 평균들의 분산 (배치 간 분산) |

세 분산 모두 **개수 그대로 나누는 분산**(`np.var`의 기본값, `ddof=0`)으로 계산한다. 이렇게 하면 배치 크기가 같을 때 `total == within + between`이 반올림 오차 안에서 정확히 성립한다. 테스트는 이 항등식, 배치 간 차이가 없을 때와 배치 안 잡음이 없을 때의 극단적인 경우, 그리고 시뮬레이션 데이터에서 각 성분이 이론값에 가까운지를 확인한다.

힌트: `axis` 인자로 배치 방향의 평균과 분산을 구한다.
