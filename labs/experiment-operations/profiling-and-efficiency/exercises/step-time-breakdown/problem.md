# step 시간 분해하기

학습 루프에 시계를 달아 step마다 두 구간의 시간을 기록했다(GPU 코드라면 각 구간 끝에서 `torch.cuda.synchronize()`를 부른 뒤 잰 값이다).

```python
steps = [
    {"step": 0, "batch_size": 64, "data_s": 0.90, "compute_s": 2.10},   # 첫 step: 초기화 비용 포함
    {"step": 1, "batch_size": 64, "data_s": 0.04, "compute_s": 0.06},
    ...
]
```

- `data_s`: 다음 batch를 기다린 시간(데이터 로딩·전처리·장치로 복사)
- `compute_s`: forward·backward·optimizer step 시간

함수 `summarize_steps(steps, warmup)`을 완성한다. 앞의 `warmup`개 step은 초기화 비용(메모리 할당, kernel 선택, 컴파일)이 섞여 있으므로 **모든 계산에서 뺀다.** 반환값은 다음 네 항목의 dict다.

| 키 | 정의 |
|---|---|
| `throughput` | 남은 step들의 (샘플 수 합) / (`data_s + compute_s`의 합). 단위: 샘플/초 |
| `data_fraction` | 남은 step들의 `data_s` 합 / (`data_s + compute_s`의 합) |
| `overlap_step_s` | 데이터 준비가 계산과 완전히 겹친다면(prefetch) 기대되는 평균 step 시간: 남은 step들의 `max(data_s, compute_s)`의 평균 |
| `overlap_speedup` | 지금의 평균 step 시간(`data_s + compute_s`의 평균) / `overlap_step_s` |

`warmup`이 step 수보다 크거나 같으면 `ValueError`를 낸다.

`overlap_speedup`은 데이터 공급을 고쳐서 얻을 수 있는 이득의 **상한**이다. 계산이 더 오래 걸리는 step에서는 데이터를 아무리 빨리 해도 계산 시간 아래로 내려갈 수 없다.
