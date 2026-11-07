# 지연 계산하는 batch generator

샘플이 하나씩 나오는 iterable(파일의 줄, 센서 스트림, 다른 generator)을 받아 크기 `batch_size`의 리스트로 묶어 내보내는 generator 함수 `batched(samples, batch_size)`를 완성한다.

```python
list(batched(range(7), 3))
# [[0, 1, 2], [3, 4, 5], [6]]
```

규칙은 다음과 같다.

- 마지막 batch는 `batch_size`보다 작을 수 있다. 빈 batch는 내보내지 않는다.
- `batch_size`가 1보다 작으면 `ValueError`를 낸다. 이 검사는 **첫 batch를 요청하기 전, 함수를 호출하는 순간** 이루어져야 한다.
- 입력을 **한꺼번에 읽지 않는다.** 끝이 없는 스트림(`itertools.count()`)에서도 앞의 몇 batch를 꺼낼 수 있어야 하고, batch 하나를 꺼낼 때 입력에서 정확히 그 batch에 필요한 만큼만 읽어야 한다.
- 입력은 리스트가 아닐 수 있다. 인덱싱(`samples[i]`)이나 `len(samples)`를 쓰지 않는다.

힌트: `iter()`로 iterator를 만든 뒤 `itertools.islice`로 필요한 만큼만 꺼낼 수 있다. generator 함수 본문은 첫 `next()` 호출 때 실행되기 시작한다는 점이 인자 검사 규칙과 어떻게 얽히는지 생각한다.
