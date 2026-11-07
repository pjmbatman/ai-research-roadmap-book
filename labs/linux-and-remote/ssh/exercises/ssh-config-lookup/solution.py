from fnmatch import fnmatchcase


def host_matches(patterns, host):
    """Host 줄의 패턴 목록이 host에 적용되는지. 부정 패턴이 맞으면 무조건 False."""
    matched = False
    for p in patterns:
        if p.startswith("!"):
            if fnmatchcase(host, p[1:]):
                return False
        elif fnmatchcase(host, p):
            matched = True
    return matched


def resolve(config_text, host):
    """~/.ssh/config 문자열에서 host에 적용되는 설정을 dict로 반환한다.

    키는 소문자. 각 키는 처음 얻은 값. identityfile만 모든 값을 순서대로 모은 리스트.
    """
    result = {}
    active = True                                   # 첫 Host 줄 이전은 모든 호스트에 적용
    for raw in config_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line.split(None, 1)[0]:           # 키=값
            key, value = line.split("=", 1)
        else:                                       # 키 값 (값이 없을 수도 있다)
            parts = line.split(None, 1)
            key, value = parts[0], parts[1] if len(parts) > 1 else ""
        key, value = key.strip().lower(), value.strip()
        if key == "host":
            active = host_matches(value.split(), host)
            continue
        if not active:
            continue
        if key == "identityfile":
            result.setdefault("identityfile", []).append(value)
        else:
            result.setdefault(key, value)           # 처음 얻은 값이 이긴다
    return result
