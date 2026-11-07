def merge_base(parents, a, b):
    """commit a와 b의 가장 좋은 공통 조상을 이름순 리스트로 반환한다.

    parents: dict[str, list[str]] — commit 이름 → parent 이름 리스트
    반환: list[str] — 가장 좋은 공통 조상들(없으면 빈 리스트)
    """
    raise NotImplementedError("조상 집합을 구하고, 공통 조상 중 가장 좋은 것만 남기세요")
