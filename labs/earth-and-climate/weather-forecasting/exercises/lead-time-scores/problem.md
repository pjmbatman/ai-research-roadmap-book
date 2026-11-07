# 리드 타임별 위도 가중 RMSE와 ACC

예보 `N`개를 리드 타임 `L`개까지 위경도 격자에서 평가한다. 함수 `scores(fc, obs, clim, lat_deg)`를 완성해 리드 타임별 `(rmse, acc)`를 반환한다.

| 인자 | shape | 설명 |
|---|---|---|
| `fc` | `(N, L, nlat, nlon)` | 예보 |
| `obs` | `(N, L, nlat, nlon)` | 같은 시각의 정답(분석장) |
| `clim` | `(nlat, nlon)` | 학습 기간에서 계산한 기후값 |
| `lat_deg` | `(nlat,)` | 칸 중심 위도 [도] |

위도 가중치를 $w_i = \cos\varphi_i / \overline{\cos\varphi}$(격자 전체 평균이 1이 되도록 정규화)로 두고, 예보 $n$, 리드 $\ell$에 대해

$$
\mathrm{RMSE}_\ell = \sqrt{\frac{1}{N}\sum_{n} \overline{w\,(f_{n\ell} - o_{n\ell})^2}}, \qquad
\mathrm{ACC}_\ell = \frac{1}{N}\sum_{n} \frac{\sum_{ij} w_i\, f'_{n\ell ij}\, o'_{n\ell ij}}{\sqrt{\sum_{ij} w_i\, f'^2_{n\ell ij}\;\sum_{ij} w_i\, o'^2_{n\ell ij}}}
$$

를 계산한다. 윗줄 $\overline{\,\cdot\,}$는 격자 칸 평균, $f' = f - c$, $o' = o - c$는 기후값 $c$에 대한 편차다.

- 반환하는 `rmse`, `acc`의 shape은 모두 `(L,)`이다.
- RMSE는 예보마다 MSE를 구해 평균한 뒤 제곱근을 취한다(예보마다 RMSE를 구해 평균하지 않는다).
- ACC는 예보마다 공간 상관을 구한 뒤 평균한다.

힌트: `w = np.cos(np.deg2rad(lat_deg))[:, None]`를 `w.mean()`으로 나누면 broadcasting으로 모든 경도에 같은 가중치가 걸린다.
