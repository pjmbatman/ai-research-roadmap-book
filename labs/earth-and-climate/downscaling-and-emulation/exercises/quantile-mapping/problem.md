# 경험적 quantile mapping

기후 모델 출력의 분포를 관측 분포에 맞추는 가장 널리 쓰이는 편향 보정은 **quantile mapping**이다. 모델 값 $x$가 과거 기간 모델 분포에서 차지하는 분위 $u = \hat F_\text{mod}(x)$를 구하고, 같은 분위의 관측 값 $\hat F_\text{obs}^{-1}(u)$로 바꾼다.

$$
\tilde{x} = \hat F_\text{obs}^{-1}\big(\hat F_\text{mod}(x)\big)
$$

함수 `quantile_map(x, model_ref, obs_ref, n_q=1001)`을 완성한다.

| 인자 | shape | 설명 |
|---|---|---|
| `x` | 임의 | 보정할 모델 값 |
| `model_ref` | `(n,)` | 보정 기간(과거)의 모델 값 |
| `obs_ref` | `(m,)` | 같은 기간의 관측 값 |
| `n_q` | 정수 | 분위 격자의 점 수 |

구현 규칙(테스트가 이 규칙을 가정한다):

1. 분위 격자 `qs = np.linspace(0, 1, n_q)`에서 `model_ref`와 `obs_ref`의 경험적 분위수 `mq`, `oq`를 `np.quantile`로 구한다.
2. `np.interp(x, mq, oq)`로 사이 값을 선형 보간한다. 따라서 `model_ref`의 범위를 벗어난 `x`는 관측의 최솟값·최댓값으로 **고정**된다.
3. 반환값의 shape은 `x`와 같다.

2번의 고정은 의도한 규칙이다. 테스트 하나가 이 성질을 확인하는데, 미래 기후에서 보정 기간보다 큰 값이 나오면 이 방법이 극값을 잘라 버린다는 것을 보이기 위해서다.

힌트: 세 줄이면 된다. 결과를 `model_ref`에 적용해 분위수가 `obs_ref`와 같아지는지 확인한다.
