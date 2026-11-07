# 비어 있는 GPU를 골라 CUDA_VISIBLE_DEVICES 만들기

scheduler 없이 여러 사람이 함께 쓰는 GPU 서버에서, 학습을 띄우기 전에 비어 있는 GPU를 확인하고 그 GPU만 보이게 한다. 두 명령의 출력을 받는다.

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
```

```text
0, GPU-aaaa, 30512, 40960
1, GPU-bbbb, 38200, 40960
2, GPU-cccc, 4, 40960
```

```text
GPU-aaaa, 5310, 30500
GPU-bbbb, 4190, 38190
```

프로세스 목록은 GPU 번호가 아니라 **uuid**로 GPU를 가리킨다는 점에 주의한다.

함수 `pick_gpus(gpu_csv, apps_csv, n, min_free_mib)`를 완성한다.

- GPU가 "비어 있다"는 것은 (1) 그 GPU에서 도는 프로세스가 하나도 없고, (2) 빈 메모리(`memory.total - memory.used`)가 `min_free_mib` 이상인 것이다. 프로세스가 없어도 메모리가 차 있으면(다른 컨테이너의 프로세스는 목록에 안 보일 수 있다) 비어 있지 않다.
- 비어 있는 GPU 중 **번호가 작은 것부터** `n`개를 골라, `CUDA_VISIBLE_DEVICES`에 넣을 문자열(예: `"2,3"`)을 반환한다.
- 비어 있는 GPU가 `n`개보다 적으면 `RuntimeError`를 낸다. 남의 GPU에 올리는 대신 기다리거나 사람에게 묻는 것이 맞다.
- 빈 줄은 무시한다. `apps_csv`가 빈 문자열이면 아무 프로세스도 없는 것이다.
