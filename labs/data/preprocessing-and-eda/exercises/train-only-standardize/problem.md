# train에서만 표준화 통계량 추정하기

표준화 $z = (x - \mu) / \sigma$의 $\mu$와 $\sigma$는 데이터에서 **추정한** 값이다. 추정에 test 데이터를 쓰면 test의 정보가 모델 입력에 섞인다. 통계량은 train에서만 추정하고, 같은 값을 train과 test에 똑같이 적용한다.

함수 `standardize_split(X_train, X_test)`를 완성한다.

| 인자·반환 | shape | 뜻 |
|---|---|---|
| `X_train` | `(N_train, D)` | 학습 데이터. `NaN`(결측)이 있을 수 있다 |
| `X_test` | `(N_test, D)` | 평가 데이터. `NaN`이 있을 수 있다 |
| 반환 `Z_train`, `Z_test` | 입력과 같음 | 표준화한 배열. `NaN`은 `NaN`으로 남는다 |
| 반환 `mean`, `std` | `(D,)` | train에서 열마다 추정한 평균과 표준편차 |

규칙:

- 평균과 표준편차는 `X_train`의 결측이 아닌 값만으로 열마다 구한다. 표준편차는 모표준편차(`ddof=0`)다.
- train에서 값이 하나뿐인 열(표준편차 0)은 0으로 나누지 않도록 `std`를 1로 둔다.
- `X_test`의 값은 통계량 추정에 쓰지 않는다.

힌트: `np.nanmean`, `np.nanstd`를 `axis=0`으로 쓴다.
