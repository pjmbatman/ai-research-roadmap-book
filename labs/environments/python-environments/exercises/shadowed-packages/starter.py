def find_shadowed(sys_path, installed):
    """여러 디렉터리에 설치되어 하나가 다른 것을 가리는 패키지를 찾는다.

    sys_path: 검색 순서대로의 디렉터리 목록
    installed: {디렉터리: {패키지 이름: 버전}}
    반환: {정규화한 이름: {"used": (버전, 디렉터리), "shadowed": [(버전, 디렉터리), ...]}}
    """
    raise NotImplementedError("sys_path 순서대로 패키지를 모으세요")
