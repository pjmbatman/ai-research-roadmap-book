# 순서를 지키는 선형 시간 중복 제거

데이터 파일 목록, 샘플 ID, 분자 식별자(SMILES 문자열)처럼 중복이 섞인 목록에서, **처음 나온 순서를 지키며** 중복을 제거하는 함수 `dedup(items)`를 완성한다.

```python
dedup(["b", "a", "b", "c", "a"])   # ["b", "a", "c"]
```

다음과 같은 구현은 결과는 맞지만 $O(n^2)$이다. `x not in out`이 리스트 `out` 전체를 훑기 때문이다.

```python
def dedup_slow(items):
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out
```

요구 사항은 다음과 같다.

- 결과는 리스트이고, 각 값이 처음 나온 순서를 지킨다.
- 원소는 hash 가능하다고 가정한다.
- 평균 $O(n)$이어야 한다. 테스트는 `==` 비교가 몇 번 일어나는지 세어, 원소 수의 몇 배를 넘으면 실패시킨다. 리스트에서 membership을 검사하면 이 조건을 통과할 수 없다.
- 입력 리스트는 바꾸지 않는다.

힌트: 이미 본 값을 set에 기록한다. `dict.fromkeys(items)`가 삽입 순서를 지킨다는 성질을 써도 된다.
