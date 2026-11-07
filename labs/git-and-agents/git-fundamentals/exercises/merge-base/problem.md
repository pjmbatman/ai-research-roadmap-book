# 두 commit의 merge-base 찾기

3-way merge는 두 branch의 끝 commit뿐 아니라 **공통 조상**(merge-base)을 기준으로 삼는다. `git merge-base A B`가 하는 일을 직접 구현한다.

commit DAG는 딕셔너리 `parents`로 주어진다. 키는 commit 이름, 값은 parent 이름의 리스트다(첫 commit은 빈 리스트, merge commit은 parent가 둘).

```python
parents = {"A": [], "B": ["A"], "C": ["B"], "D": ["B"], "E": ["D"]}
merge_base(parents, "C", "E")   # ['B']
```

함수 `merge_base(parents, a, b)`를 완성한다.

- **공통 조상**은 `a`와 `b` 양쪽에서 parent를 따라 도달할 수 있는 commit이다. 자기 자신도 조상에 포함한다(`a`가 `b`의 조상이면 답은 `a`).
- 그중 **가장 좋은 공통 조상**만 반환한다. 다른 공통 조상의 조상인 commit은 제외한다.
- 가장 좋은 공통 조상이 여러 개일 수 있다(두 branch가 서로를 번갈아 merge한 criss-cross 이력). 이름순으로 정렬한 리스트로 반환한다.
- 공통 조상이 없으면(이력이 이어지지 않은 두 저장소) 빈 리스트를 반환한다.

힌트: 먼저 한 commit의 조상 집합을 구하는 함수를 만든다. 공통 조상 집합에서, 다른 공통 조상 $y \neq x$의 조상 집합에 들어 있는 $x$를 지운다.

참고: Git의 기본 병합 전략은 가장 좋은 공통 조상이 여러 개이면 그것들을 먼저 병합해 가상의 기준 commit을 만든다.
