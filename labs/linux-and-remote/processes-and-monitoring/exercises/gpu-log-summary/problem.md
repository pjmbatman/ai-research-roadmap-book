# nvidia-smi 기록 요약하기

학습 중 GPU 상태를 1초마다 파일에 남겨 두었다.

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv,noheader,nounits -l 1 > gpu.log
```

각 줄은 다음과 같다(쉼표 뒤에 공백이 있다).

```text
2026/03/02 10:00:01.123, 0, 98, 30512, 40960
2026/03/02 10:00:01.125, 1, 0, 38200, 40960
```

함수 `summarize(text, idle_util_pct=5, held_mem_mib=1024)`를 완성해 GPU 번호마다 다음을 담은 dict를 반환한다.

```python
{0: {"mean_util": 97.5, "max_mem_mib": 30600, "idle_holding": False}, ...}
```

- `mean_util`: 사용률(%)의 평균(float). 값이 `[N/A]`인 표본은 평균에서 뺀다. 유효한 표본이 하나도 없으면 `None`.
- `max_mem_mib`: `memory.used`의 최댓값(int). `[N/A]`는 무시한다.
- `idle_holding`: `mean_util < idle_util_pct`이고 `max_mem_mib > held_mem_mib`이면 `True`. 메모리는 잡고 있는데 계산은 거의 하지 않는 GPU(2.2장 "GPU 메모리를 점유한 유령 프로세스")를 찾는 기준이다. `mean_util`이 `None`이면 `False`.

빈 줄과 `timestamp`로 시작하는 헤더 줄은 건너뛴다. 키는 GPU 번호(int)다.
