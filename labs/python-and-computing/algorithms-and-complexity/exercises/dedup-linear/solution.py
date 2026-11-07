def dedup(items):
    """처음 나온 순서를 지키며 중복을 제거한 새 리스트를 반환한다. 평균 O(n).

    items: hash 가능한 원소의 iterable
    """
    seen = set()
    out = []
    for x in items:
        if x not in seen:          # set membership: 평균 O(1)
            seen.add(x)
            out.append(x)
    return out
