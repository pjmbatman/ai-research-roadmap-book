"""3.5 본문 코드 셀. 본문의 각 코드 블록은 이 파일의 해당 셀과 같아야 하고, 출력은 이 파일을 실행한 결과다.

실행: uv run --project labs python labs/environments/docker/cells.py
"""
# %% cell 1 — tag는 움직이는 이름, digest는 내용의 해시
import hashlib

def digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()[:12] + "..."

v1 = b'{"layers": ["ubuntu-base", "python-3.12", "numpy-2.5.4"]}'
v2 = b'{"layers": ["ubuntu-base", "python-3.12", "numpy-2.5.5"]}'   # 한 글자 차이

registry = {"lab/proj:latest": digest(v1)}
print("1월 push  lab/proj:latest ->", registry["lab/proj:latest"])
registry["lab/proj:latest"] = digest(v2)                             # 같은 tag로 다시 push
print("6월 push  lab/proj:latest ->", registry["lab/proj:latest"])
print("1월 내용을 digest로 다시 계산 ->", digest(v1))

# %% cell 2 — layer cache: 어떤 줄부터 다시 빌드되는가
def rebuilt(dockerfile, changed):
    """changed 파일이 바뀌었을 때 다시 실행되는 줄. 한 줄이 무효화되면 그 뒤는 모두 다시 실행된다."""
    for i, (line, inputs) in enumerate(dockerfile):
        if inputs & changed:
            return [l for l, _ in dockerfile[i:]]
    return []

code_first = [("FROM python:3.12-slim", set()),
              ("COPY . /app", {"pyproject.toml", "uv.lock", "train.py"}),
              ("RUN uv sync --locked", set())]
lock_first = [("FROM python:3.12-slim", set()),
              ("COPY pyproject.toml uv.lock /app/", {"pyproject.toml", "uv.lock"}),
              ("RUN uv sync --locked --no-install-project", set()),
              ("COPY . /app", {"pyproject.toml", "uv.lock", "train.py"})]

for name, df in [("코드 먼저", code_first), ("lockfile 먼저", lock_first)]:
    print(f"[{name}] train.py 수정 -> 다시 실행:", rebuilt(df, {"train.py"}))

# %% cell 3 — DataLoader worker가 쓰는 shared memory 어림
B, C, H, W = 64, 3, 224, 224                    # 이미지 batch (B, C, H, W)
bytes_per_batch = B * C * H * W * 4             # float32
num_workers, prefetch_factor = 8, 2             # worker마다 미리 준비해 두는 batch 수
in_flight = num_workers * prefetch_factor

need_mib = in_flight * bytes_per_batch / 2**20
print(f"batch 하나: {bytes_per_batch / 2**20:.1f} MiB")
print(f"동시에 대기하는 batch: {in_flight}개 -> 약 {need_mib:.0f} MiB")
print(f"Docker 기본 /dev/shm 64 MiB의 {need_mib / 64:.1f}배")
