def _resolve(path, links):
    """디렉터리 링크를 따라간 실제 경로. 디렉터리 단위로 앞부분이 일치할 때만 바꾼다."""
    for src, dst in links.items():
        src, dst = src.rstrip("/"), dst.rstrip("/")
        if path == src or path.startswith(src + "/"):
            return dst + path[len(src):]
    return path


def lookup(name, path_var, files, links):
    """PATH(또는 LD_LIBRARY_PATH) 규칙으로 name을 찾는다.

    name: 파일 이름, path_var: "a:b:c" 형식의 디렉터리 목록
    files: 존재하는 파일의 절대 경로 집합, links: {링크 디렉터리: 실제 디렉터리}
    반환: (찾은 경로, 실제 경로) 또는 None
    """
    for d in path_var.split(":"):
        if not d:
            continue
        candidate = d.rstrip("/") + "/" + name
        real = _resolve(candidate, links)
        if real in files:
            return candidate, real
    return None
