def _norm(name):
    return name.lower().replace("_", "-")


def find_shadowed(sys_path, installed):
    """여러 디렉터리에 설치되어 하나가 다른 것을 가리는 패키지를 찾는다.

    sys_path: 검색 순서대로의 디렉터리 목록
    installed: {디렉터리: {패키지 이름: 버전}}
    반환: {정규화한 이름: {"used": (버전, 디렉터리), "shadowed": [(버전, 디렉터리), ...]}}
    """
    found = {}                                   # 이름 -> [(버전, 디렉터리), ...] 검색 순서대로
    for d in sys_path:
        for name, version in installed.get(d, {}).items():
            found.setdefault(_norm(name), []).append((version, d))
    return {
        name: {"used": hits[0], "shadowed": hits[1:]}
        for name, hits in found.items()
        if len(hits) > 1
    }
