# epoch별 batch 인덱스와 worker 배분

DataLoader는 데이터를 직접 읽기 전에 **인덱스**부터 정한다. sampler가 한 epoch 동안 방문할 표본 순서를 만들고, batch sampler가 그 순서를 batch 크기로 자르고, 각 batch를 worker 프로세스에 나눠 맡긴다. 이 흐름을 NumPy로 구현한다.

1. `epoch_batches(n, batch_size, seed, epoch, shuffle=True, drop_last=False)`
   - 표본 `0, ..., n-1`을 한 번씩 방문하는 batch 목록(정수 list의 list)을 반환한다.
   - `shuffle=True`이면 순서를 섞는다. 섞는 순서는 `(seed, epoch)`만으로 정해져야 한다. 같은 `(seed, epoch)`이면 항상 같은 결과, `epoch`이 다르면 (거의 항상) 다른 순서다. 힌트: `np.random.default_rng([seed, epoch])`.
   - `shuffle=False`이면 `0, 1, 2, ...` 순서다.
   - 마지막 batch가 `batch_size`보다 작으면 `drop_last=True`일 때 버리고, `False`일 때 남긴다.
2. `assign_to_workers(batches, num_workers)`
   - batch 목록을 worker들에게 **차례로 돌아가며** 배분한다. worker `k`는 batch `k, k + W, k + 2W, ...`를 맡는다(`W = num_workers`). map-style Dataset에서 PyTorch DataLoader가 batch를 worker에 나누는 기본 방식이다.
   - 반환: 길이 `num_workers`인 list. `k`번째 원소는 worker `k`가 맡은 batch들의 list.

worker들이 batch를 동시에 만들어도 DataLoader는 batch를 원래 순서대로 돌려준다(집필 시점 PyTorch의 기본값 `in_order=True`). 그래서 worker 수를 바꿔도 학습에 들어가는 batch 순서는 sampler가 정한 순서와 같다. 다만 worker 안에서 쓰는 난수(데이터 증강)는 worker마다 시드를 따로 정해야 한다(본문 참고).
