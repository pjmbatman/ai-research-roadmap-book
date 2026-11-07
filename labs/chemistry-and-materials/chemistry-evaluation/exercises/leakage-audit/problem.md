# 분할 간 근접 중복 누수 감사

분자 지문 행렬 `X`(비트, `(N, L)`)와 train·test 인덱스가 주어진다. test 분자 중 train에 **근접 중복**(Tanimoto 유사도가 임계 이상인 분자)을 가진 것의 비율을 재는 함수 `leaked_fraction(X, train_idx, test_idx, threshold)`를 완성한다.

두 비트 벡터 $\mathbf{a}, \mathbf{b}$의 Tanimoto 유사도는

$$
T(\mathbf{a}, \mathbf{b}) = \frac{\lvert \mathbf{a} \wedge \mathbf{b} \rvert}{\lvert \mathbf{a} \vee \mathbf{b} \rvert}
$$

다. 각 test 분자에 대해 모든 train 분자와의 유사도 중 **최댓값**을 구하고, 그 값이 `threshold` 이상이면 "누수된" 분자로 센다. 반환값은 누수된 test 분자의 비율(0~1 스칼라)이다.

| 인자 | 의미 |
|---|---|
| `X` | `(N, L)` 비트 지문(0/1 또는 bool) |
| `train_idx`, `test_idx` | train·test 분자의 인덱스 배열 |
| `threshold` | 근접 중복으로 볼 Tanimoto 임계 |

테스트는 다음을 확인한다.

- test 분자가 train에 완전히 같은 복사본을 가지면 그 분자는 누수로 센다(유사도 1).
- 임계를 높이면 누수 비율이 단조로 줄어든다(늘지 않는다).
- train과 겹치는 비트가 전혀 없으면 누수 비율은 0이다.

힌트: 교집합은 `X[test] @ X[train].T`(float)로, 합집합은 각 행의 비트 수 합에서 교집합을 빼서 구한다.
