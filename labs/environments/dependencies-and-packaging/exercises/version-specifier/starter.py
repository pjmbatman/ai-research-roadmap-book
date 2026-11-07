def satisfies(version, spec):
    """version(예 "1.26.4")이 spec(예 ">=1.23,<2.6")의 모든 조건을 만족하면 True.

    지원: ==, !=, >=, <=, >, <, ~=, 그리고 ==X.Y.* / !=X.Y.*
    버전은 점으로 구분한 정수만(사전 배포 rc·dev 등은 다루지 않는다).
    """
    raise NotImplementedError("쉼표로 나눈 각 조건을 검사하세요")
