# stride로 view와 contiguous 판정하기

텐서는 1차원 메모리 위에 shape과 stride(각 축으로 한 칸 갈 때 건너뛰는 원소 수)를 얹은 것이다. `transpose`·`permute`는 메모리를 옮기지 않고 stride만 바꾼다. 세 함수를 완성한다. stride는 바이트가 아니라 **원소 수** 단위다.

1. `row_major_strides(shape)` — 새로 만든 행 우선(C 순서) 텐서의 stride. 마지막 축의 stride는 1이고, 앞 축의 stride는 뒤 축들의 길이를 곱한 값이다. 예: `(2, 3, 4)` → `(12, 4, 1)`, `()` → `()`.
2. `permute_strides(shape, strides, dims)` — `permute(dims)` 결과의 `(shape, strides)`. 축의 순서만 바꾼다. 예: shape `(2, 3)`, strides `(3, 1)`, dims `(1, 0)` → `((3, 2), (1, 3))`.
3. `is_contiguous(shape, strides)` — 이 shape·stride가 행 우선으로 빈틈없이 놓인 배치와 같은지. **길이가 1인 축의 stride는 무시한다**(그 축으로는 움직일 일이 없기 때문이다). PyTorch의 `is_contiguous()`, NumPy의 `flags["C_CONTIGUOUS"]`와 같은 판정이다.

각 함수는 tuple을 반환한다(`is_contiguous`는 bool). 테스트는 NumPy 배열의 실제 stride(`arr.strides`를 원소 크기로 나눈 값)와 비교한다. `view`는 contiguous가 아닌 텐서에서 실패할 수 있고, 그때 `reshape`는 복사를 한다는 본문의 내용을 이 판정으로 설명할 수 있다.
