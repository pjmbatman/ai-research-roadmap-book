def mode_to_octal(s):
    """ls -l의 권한 문자열(9 또는 10글자)을 chmod 8진수 문자열로 바꾼다.

    예: "rw-r-----" → "640", "drwxrwsr-x" → "2775"
    길이가 9나 10이 아니면 ValueError.
    """
    if len(s) == 10:
        s = s[1:]                                  # 파일 종류 글자를 뗀다
    if len(s) != 9:
        raise ValueError(f"권한 문자열은 9 또는 10글자여야 합니다: {s!r}")

    special_letters = ["sS", "sS", "tT"]           # 소유자·그룹·기타 실행 자리의 특수 표시
    special_values = [4, 2, 1]                     # setuid, setgid, sticky
    special = 0
    digits = []
    for k in range(3):
        r, w, x = s[3 * k:3 * k + 3]
        value = 4 * (r == "r") + 2 * (w == "w")
        if x == "x" or x == special_letters[k][0]:  # 소문자 s/t는 실행 비트도 켜져 있다
            value += 1
        if x in special_letters[k]:
            special += special_values[k]
        digits.append(str(value))
    body = "".join(digits)
    return f"{special}{body}" if special else body
