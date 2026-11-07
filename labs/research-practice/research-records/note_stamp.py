"""연구 노트 항목에 붙일 '도장'(날짜·커밋·작업 트리 상태·설정 hash)을 출력한다.

실행: uv run --project labs python labs/research-practice/research-records/note_stamp.py [설정 파일 ...]
설정 파일은 JSON 또는 TOML. 키 순서·공백과 무관한 hash를 만들기 위해 정규화한 JSON을 hash한다.
"""
import datetime
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def config_hash(cfg: dict) -> str:
    """키 순서와 공백에 무관한 설정 hash(앞 12자리)."""
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)


def stamp(config_paths):
    commit = git("rev-parse", "--short", "HEAD") or "git 없음"
    dirty = git("status", "--porcelain", "--untracked-files=no")
    lines = [f"- 시각: {datetime.datetime.now().astimezone().isoformat(timespec='minutes')}",
             f"- 커밋: {commit}" + (" (커밋하지 않은 변경 있음!)" if dirty else "")]
    for p in map(Path, config_paths):
        lines.append(f"- 설정: {p} sha256:{config_hash(load(p))}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(stamp(sys.argv[1:]))
