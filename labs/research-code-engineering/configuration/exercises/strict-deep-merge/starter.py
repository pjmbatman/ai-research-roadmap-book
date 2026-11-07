import copy


def deep_merge(base: dict, override: dict) -> dict:
    """override를 base 위에 깊게 병합한 새 dict를 반환한다.

    base에 없는 키 → KeyError(점 경로), dict와 dict가 아닌 값의 충돌 → TypeError(점 경로).
    base와 override는 바꾸지 않고, 반환값은 입력과 중첩 객체를 공유하지 않는다.
    """
    raise NotImplementedError("깊은 병합을 구현하세요")
