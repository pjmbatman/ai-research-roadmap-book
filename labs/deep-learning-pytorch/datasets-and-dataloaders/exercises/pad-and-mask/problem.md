# 가변 길이 시계열을 padding과 mask로 batch화하기

길이가 다른 시계열 $\mathbf{x}_i \in \mathbb{R}^{T_i \times C}$들을 하나의 배열로 묶으려면 가장 긴 길이 $T_{\max}$에 맞춰 뒤를 채워야(padding) 한다. 채운 값은 데이터가 아니므로, 어디가 실제 값인지 알려 주는 mask를 함께 만들고 손실과 metric은 mask가 `True`인 곳에서만 계산한다. NumPy로 두 함수를 완성한다.

1. `pad_batch(seqs, pad_value=0.0)`
   - `seqs`: 길이가 다른 2차원 배열 `(T_i, C)`의 list. 모든 배열의 `C`는 같다.
   - 반환: `(x, mask, lengths)`
     - `x`: `(B, T_max, C)` float 배열. 앞쪽은 원래 값, 뒤쪽은 `pad_value`.
     - `mask`: `(B, T_max)` bool 배열. 실제 값이면 `True`.
     - `lengths`: `(B,)` 정수 배열.
   - 입력 배열을 바꾸지 않는다.
2. `masked_mse(pred, target, mask)`
   - `pred`, `target`: `(B, T, C)`, `mask`: `(B, T)`.
   - mask가 `True`인 위치의 모든 원소에 대한 제곱오차의 평균을 반환한다(float). padding 위치의 값이 무엇이든 결과가 같아야 한다.
   - mask가 모두 `False`이면 `ValueError`를 낸다(0으로 나누는 대신 문제를 드러낸다).

이렇게 만든 손실은 **원소 단위 평균**이다. 긴 시계열이 짧은 시계열보다 더 큰 비중을 가진다. 표본마다 같은 비중을 원하면 표본별 평균을 먼저 구한 뒤 평균한다(본문 참고). 어느 쪽을 쓸지는 설계 선택이므로 코드와 보고서에 밝힌다.
