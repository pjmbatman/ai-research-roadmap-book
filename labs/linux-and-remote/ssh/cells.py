"""2.3 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/linux-and-remote/ssh/cells.py
"""
# %% cell 1 — 키 파일 권한 검사: 그룹·기타에게 열린 비트가 있는가
def too_open(mode):
    return mode & 0o077 != 0                  # 그룹(070)·기타(007) 비트 중 하나라도 켜져 있으면

for name, mode in [("id_ed25519", 0o600), ("id_ed25519", 0o644), ("id_ed25519", 0o400),
                   ("id_ed25519.pub", 0o644), ("~/.ssh", 0o700), ("~/.ssh", 0o775)]:
    print(f"{name:15s} {mode:04o}  그룹·기타 비트 {mode & 0o077:03o}  {'열림' if too_open(mode) else '닫힘'}")

# %% cell 2 — ~/.ssh/config: 각 설정은 '처음 얻은 값'이 이긴다
from fnmatch import fnmatch

config = [                                    # (Host 패턴, {설정}) — 파일에 적힌 순서
    ("*",    {"user": "kim", "serveraliveinterval": "60"}),
    ("gpu1", {"hostname": "10.0.3.21", "user": "lab", "proxyjump": "bastion"}),
]

def resolve(host, blocks):
    result = {}
    for pattern, options in blocks:
        if fnmatch(host, pattern):
            for key, value in options.items():
                result.setdefault(key, value)  # 이미 정해진 값은 덮어쓰지 않는다
    return result

print("Host * 가 먼저:", resolve("gpu1", config))
print("gpu1 이 먼저: ", resolve("gpu1", config[::-1]))
