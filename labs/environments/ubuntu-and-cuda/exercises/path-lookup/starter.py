def lookup(name, path_var, files, links):
    """PATH(또는 LD_LIBRARY_PATH) 규칙으로 name을 찾는다.

    name: 파일 이름, path_var: "a:b:c" 형식의 디렉터리 목록
    files: 존재하는 파일의 절대 경로 집합, links: {링크 디렉터리: 실제 디렉터리}
    반환: (찾은 경로, 실제 경로) 또는 None
    """
    raise NotImplementedError("앞에서부터 처음 찾은 파일을 반환하세요")
