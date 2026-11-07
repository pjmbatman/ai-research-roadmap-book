def merge_base(parents, a, b):
    """commit a와 b의 가장 좋은 공통 조상을 이름순 리스트로 반환한다.

    parents: dict[str, list[str]] — commit 이름 → parent 이름 리스트
    반환: list[str] — 가장 좋은 공통 조상들(없으면 빈 리스트)
    """

    def ancestors(c):
        seen, stack = set(), [c]
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(parents[x])
        return seen

    common = ancestors(a) & ancestors(b)
    best = {x for x in common if not any(x in ancestors(y) for y in common if y != x)}
    return sorted(best)
