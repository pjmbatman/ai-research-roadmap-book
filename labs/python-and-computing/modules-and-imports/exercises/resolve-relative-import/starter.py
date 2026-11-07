def resolve(name: str, package: str | None) -> str:
    """상대 import 이름 name을 package(__package__) 기준의 절대 모듈 이름으로 바꾼다.

    점이 없으면 name을 그대로 반환한다.
    package가 None이거나 비었거나, 최상위보다 위로 올라가면 ImportError.
    """
    raise NotImplementedError("절대 이름을 계산하세요")
