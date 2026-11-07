def mode_to_octal(s):
    """ls -l의 권한 문자열(9 또는 10글자)을 chmod 8진수 문자열로 바꾼다.

    예: "rw-r-----" → "640", "drwxrwsr-x" → "2775"
    길이가 9나 10이 아니면 ValueError.
    """
    raise NotImplementedError("권한 문자열을 8진수로 바꾸세요")
