# D-최적 설계를 찾는 교환 알고리즘

회귀 모델 $\mathbf{y} = \mathbf{F}(\mathbf{x})\,\boldsymbol{\beta} + \boldsymbol{\varepsilon}$의 설계 행렬 $\mathbf{F}$는 실험 조건마다 한 행이다. D-최적 설계는 $\log\det(\mathbf{F}^\top\mathbf{F})$를 최대화하는 실험 조건 $n$개를 고른다.

함수 `d_optimal(features, cand, n, rng, sweeps)`를 완성한다.

| 인자 | 뜻 |
|---|---|
| `features` | 조건 배열 `(n,)`을 받아 설계 행렬 `(n, p)`를 돌려주는 함수 |
| `cand` | 후보 실험 조건 `(C,)` |
| `n` | 실험 수 |
| `rng` | `np.random.Generator`. 초기 설계를 `rng.choice(cand, size=n)`으로 뽑는다 |
| `sweeps` | 최대 반복 횟수 |

알고리즘:

1. 무작위 초기 설계에서 시작한다.
2. 한 번의 sweep에서 각 위치 $i$의 조건을 후보 전체로 바꿔 보고, $\log\det$가 가장 커지는 후보가 현재보다 좋으면 바꾼다.
3. 한 sweep 동안 아무것도 바뀌지 않으면 멈춘다.
4. 정렬한 설계 `(n,)`을 반환한다.

$\mathbf{F}^\top\mathbf{F}$가 특이하면 $\log\det$는 $-\infty$로 취급한다(`np.linalg.slogdet`의 부호가 0 이하인 경우).

테스트는 다음을 확인한다.

- 직선 모델(실험 4번)에서 양 끝 $\{-1, -1, 1, 1\}$을 찾는지
- 2차 모델(실험 6번)에서 $\{-1, -1, 0, 0, 1, 1\}$을 찾는지
- 결과의 $\log\det$가 같은 실험 수의 등간격 설계보다 크거나 같은지

힌트: 본문 셀 4와 같은 알고리즘이다. 특이 행렬 처리만 덧붙인다.
