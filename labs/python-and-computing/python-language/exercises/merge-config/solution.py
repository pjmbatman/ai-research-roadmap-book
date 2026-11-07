import copy


def merge_config(base: dict, overrides: dict) -> dict:
    """base에 overrides를 재귀적으로 덮어쓴 새 dict를 반환한다.

    두 값이 모두 dict이면 재귀적으로 병합하고, 그 밖에는 overrides의 값이 이긴다.
    base와 overrides는 바꾸지 않고, 반환값은 입력과 객체를 공유하지 않는다.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
