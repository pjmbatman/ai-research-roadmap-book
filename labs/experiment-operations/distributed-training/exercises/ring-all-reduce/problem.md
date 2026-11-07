# ring all-reduce 시뮬레이션

데이터 병렬 학습은 매 step 모든 GPU의 gradient를 더한다(all-reduce). 가장 단순한 방법은 모두가 rank 0에게 보내고 rank 0이 합을 다시 모두에게 보내는 것이지만, 그러면 rank 0 혼자 GPU 수에 비례하는 양을 주고받아 병목이 된다. **ring all-reduce**는 GPU들을 고리로 잇고 배열을 $W$개 청크로 나누어, 모든 GPU가 같은 양을 주고받게 한다.

함수 `ring_all_reduce(arrays)`를 완성한다. `arrays`는 worker $W$개의 1차원 배열 리스트(모두 같은 shape `(n,)`)이고, 반환은 `(results, sent_bytes)`다.

- `results`: 길이 $W$의 리스트. 모든 원소가 `sum(arrays)`와 같아야 한다.
- `sent_bytes`: 길이 $W$의 리스트. worker마다 **보낸** byte 수(`ndarray.nbytes`의 합).

알고리즘은 두 단계다. 배열을 `np.array_split(a, W)`로 청크 $W$개로 나누고, 각 단계에서 worker $i$는 이웃 $(i+1) \bmod W$에게만 청크 하나를 보낸다. 한 단계의 모든 전송은 **동시에** 일어난다고 본다(보내기 전에 보낼 데이터를 복사해 둔다).

1. **reduce-scatter** ($W-1$단계): 단계 $s$에서 worker $i$는 청크 $(i - s) \bmod W$를 보내고, 받은 worker는 자기 청크에 **더한다**. 끝나면 worker $i$는 청크 $(i+1) \bmod W$의 전체 합을 가진다.
2. **all-gather** ($W-1$단계): 단계 $s$에서 worker $i$는 청크 $(i + 1 - s) \bmod W$를 보내고, 받은 worker는 자기 청크를 그것으로 **바꾼다**.

worker 하나가 보내는 양은 청크 $2(W-1)$개, 즉 배열 크기의 $2(W-1)/W$배다. $W$가 커져도 2배를 넘지 않는다. 이것이 GPU 수가 늘어도 ring all-reduce의 GPU당 통신량이 거의 일정한 이유다.

테스트는 모든 worker의 결과가 합과 같은지, 각 worker가 보낸 byte가 $2(W-1)/W$ 규칙을 따르는지(모두 rank 0에게 보내는 방식은 이 검사를 통과하지 못한다), 입력을 바꾸지 않는지, $W = 1$과 $n$이 $W$로 나누어떨어지지 않는 경우를 확인한다.

힌트: 먼저 $W = 3$, $n = 3$(청크 하나에 원소 하나)으로 각 단계에서 누가 무엇을 가지는지 표로 그려 본다.
