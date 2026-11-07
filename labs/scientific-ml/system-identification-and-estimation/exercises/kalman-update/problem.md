# 칼만 필터의 갱신 단계

예측 단계가 끝난 사전분포 $\mathcal{N}(\mathbf{m}^-, \mathbf{P}^-)$와 새 측정 $\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{v}$, $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$가 주어졌다. 갱신 단계는 다음과 같다.

$$
\mathbf{S} = \mathbf{H}\mathbf{P}^-\mathbf{H}^\top + \mathbf{R}, \qquad
\mathbf{K} = \mathbf{P}^-\mathbf{H}^\top\mathbf{S}^{-1},
$$

$$
\mathbf{m} = \mathbf{m}^- + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{m}^-), \qquad
\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-.
$$

함수 `kf_update(m, P, y, H, R)`를 완성해 `(m_new, P_new)`를 반환한다.

| 인자 | shape |
|---|---|
| `m` | `(n,)` |
| `P` | `(n, n)` |
| `y` | `(k,)` |
| `H` | `(k, n)` |
| `R` | `(k, k)` |

테스트는 다음을 확인한다.

- 11.4장의 가우시안 조건부분포 공식으로 계산한 결과와 같다.
- 측정 잡음이 매우 크면 갱신이 거의 일어나지 않는다.
- 갱신된 공분산은 대칭이고, 측정한 방향의 분산은 줄어든다.

힌트: $\mathbf{S}^{-1}$을 직접 구하지 말고 `np.linalg.solve(S, ...)`를 쓴다. $\mathbf{K} = \mathbf{P}^-\mathbf{H}^\top\mathbf{S}^{-1}$이고 $\mathbf{S}$가 대칭이므로 $\mathbf{K}^\top = \mathbf{S}^{-1}\mathbf{H}\mathbf{P}^-$다.
