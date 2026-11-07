# 결측이 있는 격자의 위도 가중 평균

위경도 격자에서 칸의 면적은 $\cos\varphi$($\varphi$는 칸 중심의 위도)에 비례한다. 함수 `weighted_mean(field, lat_deg)`를 완성해 마지막 두 축(위도, 경도)에 대한 면적 가중 평균을 반환한다.

| 인자 | shape | 설명 |
|---|---|---|
| `field` | `(..., nlat, nlon)` | 격자 값. 결측은 `NaN` |
| `lat_deg` | `(nlat,)` | 칸 중심 위도 [도] |

- 앞쪽 축(시간 등)은 그대로 남긴다. 반환 shape은 `field.shape[:-2]`이다(2차원 입력이면 스칼라).
- 결측 칸은 분자와 분모 **모두**에서 뺀다. 가중치를 남은 칸에 대해 다시 정규화한다는 뜻이다.
- 모든 칸이 결측이면 `NaN`을 반환한다.

$$
\bar{x} = \frac{\sum_{i,j} m_{ij}\cos\varphi_i\, x_{ij}}{\sum_{i,j} m_{ij}\cos\varphi_i}, \qquad m_{ij} = \begin{cases}1 & x_{ij}\text{가 유효}\\ 0 & x_{ij}\text{가 결측}\end{cases}
$$

힌트: `np.isnan`으로 마스크를 만들고, `np.where`로 결측을 0으로 바꾼 값과 마스크를 곱한 가중치를 함께 더한다.
