"""4.1 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/git-and-agents/git-fundamentals/cells.py
"""
# %% cell 1 — 내용 주소: blob id 계산
import hashlib

def blob_id(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()     # "blob <바이트 수>" + NUL
    return hashlib.sha1(header + content).hexdigest()

print(blob_id(b"hello\n"))
print(blob_id(b"hello!\n"))
print(blob_id(b"lr = 0.001\n") == blob_id(b"lr = 0.001\n"))

# %% cell 2 — commit DAG: 조상 집합과 fast-forward 판정
parents = {"A": [], "B": ["A"], "C": ["B"], "F": ["C"],   # main
           "D": ["B"], "E": ["D"]}                         # feature

def ancestors(c):
    """c에서 parent를 따라 도달할 수 있는 모든 commit (c 포함)."""
    seen, stack = set(), [c]
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack.extend(parents[x])
    return seen

def is_ancestor(a, b):
    return a in ancestors(b)

print("feature만 있는 commit:", sorted(ancestors("E") - ancestors("F")))
print("main만 있는 commit:   ", sorted(ancestors("F") - ancestors("E")))
print("main에서 merge feature가 fast-forward?", is_ancestor("F", "E"))
parents["M"] = ["F", "E"]                                  # merge commit
print("merge 뒤 feature에서 merge main이 fast-forward?", is_ancestor("E", "M"))

# %% cell 3 — 3-way merge 판정 (줄 단위로 단순화)
def merge3(base, ours, theirs):
    out, conflicts = [], 0
    for b, o, t in zip(base, ours, theirs):
        if o == t:
            out.append(o)                 # 양쪽이 같다
        elif o == b:
            out.append(t)                 # 우리는 그대로, 상대만 바꿨다
        elif t == b:
            out.append(o)                 # 상대는 그대로, 우리만 바꿨다
        else:
            out.append(f"<<< {o} | {b} | {t} >>>")
            conflicts += 1
    return out, conflicts

base   = ["lr = 1e-3", "batch = 32", "seed = 0", "epochs = 10"]
ours   = ["lr = 3e-4", "batch = 32", "seed = 0", "epochs = 10"]
theirs = ["lr = 1e-3", "batch = 64", "seed = 0", "epochs = 20"]
print(merge3(base, ours, theirs))
theirs[0] = "lr = 1e-2"
print(merge3(base, ours, theirs))
