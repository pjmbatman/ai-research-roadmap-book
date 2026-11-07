import copy


def deep_merge(base: dict, override: dict, _path: str = "") -> dict:
    """override를 base 위에 깊게 병합한 새 dict를 반환한다.

    base에 없는 키 → KeyError(점 경로), dict와 dict가 아닌 값의 충돌 → TypeError(점 경로).
    base와 override는 바꾸지 않고, 반환값은 입력과 중첩 객체를 공유하지 않는다.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        path = f"{_path}.{key}" if _path else key
        if key not in out:
            raise KeyError(path)
        old_is_dict, new_is_dict = isinstance(out[key], dict), isinstance(value, dict)
        if old_is_dict != new_is_dict:
            raise TypeError(f"{path}: 섹션(dict)과 값을 서로 덮어쓸 수 없습니다")
        if new_is_dict:
            out[key] = deep_merge(out[key], value, path)
        else:
            out[key] = copy.deepcopy(value)
    return out
