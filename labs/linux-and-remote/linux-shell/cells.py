"""2.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linux-and-remote/linux-shell/cells.py
"""
# %% cell 1 — 권한 문자열을 8진수로
def perm_to_octal(s):
    """'rwxr-x---' 같은 9글자 권한 문자열을 '750' 같은 8진수 문자열로 바꾼다."""
    digits = []
    for i in range(0, 9, 3):                 # 소유자, 그룹, 기타 순
        r, w, x = s[i:i + 3]
        digits.append(str(4 * (r == "r") + 2 * (w == "w") + 1 * (x == "x")))
    return "".join(digits)

for s in ["rwxr-xr-x", "rw-r--r--", "rw-------", "rwxrwx---"]:
    print(s, "→", perm_to_octal(s))

# %% cell 2 — umask가 새 파일의 권한을 정하는 방식
def apply_umask(base, umask):
    return base & ~umask                      # umask에 켜진 비트를 끈다

for umask in [0o022, 0o002, 0o077]:
    f = apply_umask(0o666, umask)             # 새 파일의 기본 요청
    d = apply_umask(0o777, umask)             # 새 디렉터리의 기본 요청
    print(f"umask {umask:03o}  파일 {f:03o}  디렉터리 {d:03o}")

# %% cell 3 — PATH 순서와 '어느 python이 실행되는가'
installed = {                                 # 디렉터리 → 들어 있는 실행 파일(가정)
    "/home/kim/.local/bin": {"jupyter"},
    "/opt/conda/envs/proj/bin": {"python", "python3", "pip", "jupyter"},
    "/usr/bin": {"python3", "ls"},
}

def which(cmd, path):
    for d in path.split(":"):                 # 앞에서부터 찾아 처음 나온 것을 쓴다
        if cmd in installed.get(d, set()):
            return f"{d}/{cmd}"
    return None

path_a = "/opt/conda/envs/proj/bin:/home/kim/.local/bin:/usr/bin"
path_b = "/home/kim/.local/bin:/usr/bin:/opt/conda/envs/proj/bin"
for cmd in ["python3", "jupyter", "pip"]:
    print(f"{cmd:8s} A: {which(cmd, path_a):34s} B: {which(cmd, path_b)}")
