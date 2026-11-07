from fnmatch import fnmatchcase


def resolve(config_text, host):
    """~/.ssh/config 문자열에서 host에 적용되는 설정을 dict로 반환한다.

    키는 소문자. 각 키는 처음 얻은 값. identityfile만 모든 값을 순서대로 모은 리스트.
    """
    raise NotImplementedError("Host 블록을 위에서부터 읽으며 설정을 모으세요")
