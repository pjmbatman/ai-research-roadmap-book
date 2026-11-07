import inspect


def doc_drift(func):
    """NumPy 스타일 docstring의 Parameters 절과 func의 signature를 비교한다.

    반환: {"missing": signature에만 있는 인자, "stale": 문서에만 있는 인자}, 각각 정렬된 리스트
    """
    raise NotImplementedError("signature의 인자 이름과 Parameters 절의 이름을 각각 집합으로 만들어 비교하세요")
