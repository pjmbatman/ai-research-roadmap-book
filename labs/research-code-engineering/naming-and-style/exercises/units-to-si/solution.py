import numpy as np

# 접미사 → (바꾼 접미사, 배율, 더할 값)
UNITS = {
    "mv": ("v", 1e-3, 0.0), "kv": ("v", 1e3, 0.0), "v": ("v", 1.0, 0.0),
    "ma": ("a", 1e-3, 0.0), "a": ("a", 1.0, 0.0),
    "ms": ("s", 1e-3, 0.0), "min": ("s", 60.0, 0.0), "h": ("s", 3600.0, 0.0), "s": ("s", 1.0, 0.0),
    "mah": ("ah", 1e-3, 0.0), "ah": ("ah", 1.0, 0.0),
    "c": ("k", 1.0, 273.15), "k": ("k", 1.0, 0.0),
}


def to_si(record: dict, passthrough=()) -> dict:
    """이름의 단위 접미사를 읽어 값을 SI 계열 단위(v, a, s, ah, k)로 바꾼 새 dict를 반환한다.

    record: 이름 → 숫자 또는 NumPy 배열. 바꾸지 않는다.
    passthrough: 단위가 없어 그대로 둘 이름들
    알 수 없는 접미사나 변환 후 이름 충돌은 ValueError.
    """
    out = {}
    for name, value in record.items():
        if name in passthrough:
            new_name, new_value = name, value
        else:
            stem, sep, suffix = name.rpartition("_")
            if not sep or suffix not in UNITS:
                raise ValueError(f"{name!r}: 단위 접미사를 알 수 없습니다")
            si_suffix, scale, offset = UNITS[suffix]
            new_name = f"{stem}_{si_suffix}"
            new_value = np.asarray(value, dtype=float) * scale + offset
            if new_value.ndim == 0:
                new_value = float(new_value)
        if new_name in out:
            raise ValueError(f"{name!r}: 변환 후 이름 {new_name!r}이 겹칩니다")
        out[new_name] = new_value
    return out
