# roofline으로 실행 시간의 하한과 병목 판정하기

연산 하나가 장치에서 걸리는 시간은 두 가지 이유로 아래로 제한된다. 계산을 다 하는 데 걸리는 시간과, 필요한 데이터를 메모리에서 다 읽고 쓰는 데 걸리는 시간이다. 두 일이 완전히 겹쳐 진행된다고 가정하면(roofline 모형) 실행 시간의 하한은

$$
t \ge \max\left(\frac{\text{FLOP}}{P_{\text{peak}}},\ \frac{\text{bytes}}{B}\right)
$$

이다. 여기서 $P_{\text{peak}}$는 최대 연산 속도(FLOP/s), $B$는 메모리 대역폭(byte/s)이다. 두 항 중 큰 쪽이 그 연산의 병목이다.

함수 `roofline(flops, bytes_moved, peak_flops_per_s, bandwidth_bytes_per_s)`를 완성해 다음 세 값을 tuple로 반환한다.

1. `time_s`: 위 식의 하한(초)
2. `intensity`: 연산 강도 $I = \text{FLOP} / \text{bytes}$ (FLOP/byte)
3. `bound`: 메모리 시간이 연산 시간보다 **크면** `"memory"`, 아니면 `"compute"`

입력 검사도 한다.

- `flops`와 `bytes_moved`는 0 이상이어야 하고, 둘 다 0이면 안 된다.
- `peak_flops_per_s`와 `bandwidth_bytes_per_s`는 0보다 커야 한다.
- 조건을 어기면 `ValueError`를 낸다.
- `bytes_moved`가 0이면 연산 강도는 `float("inf")`로 둔다.

테스트는 벡터 덧셈, 행렬 곱 같은 실제 연산의 FLOP와 byte를 직접 세어 판정이 맞는지 확인한다. 예를 들어 `float32` 벡터 $n$개짜리 덧셈 $\mathbf{z} = \mathbf{x} + \mathbf{y}$는 덧셈 $n$번, 메모리 이동 $3 \times 4n$ byte(두 벡터 읽기, 한 벡터 쓰기)다.
