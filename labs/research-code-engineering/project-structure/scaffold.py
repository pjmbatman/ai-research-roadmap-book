"""6.1 실습: 연구 프로젝트 템플릿을 만든다.

실행 예: uv run --project labs python labs/research-code-engineering/project-structure/scaffold.py /tmp/demo --name soh_model

표준 디렉터리와 최소 파일(진입점·설정·smoke test)을 만들고 트리를 출력한다.
이미 있는 파일은 덮어쓰지 않는다.
"""
import argparse
from pathlib import Path

FILES = {
    "README.md": "# {name}\n\n목적 · 설치 · 데이터 준비 · 실행 · 결과 재현을 여기에 쓴다.\n",
    "pyproject.toml": (
        '[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n\n'
        '[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n'
    ),
    ".gitignore": "data/raw/\ndata/processed/\noutputs/\n__pycache__/\n",
    "configs/default.toml": "[data]\nroot = \"data/processed\"\n\n[model]\nwidth = 64\n\n[train]\nlr = 1e-3\nseed = 0\n",
    "src/{name}/__init__.py": "",
    "src/{name}/data/__init__.py": "",
    "src/{name}/models/__init__.py": "",
    "src/{name}/training/__init__.py": "",
    "src/{name}/evaluation/__init__.py": "",
    "src/{name}/utils/__init__.py": "",
    "scripts/train.py": "\"\"\"학습 진입점: 설정을 읽고 {name}.training을 호출한다.\"\"\"\n",
    "scripts/eval.py": "\"\"\"평가 진입점: checkpoint와 데이터만으로 {name}.evaluation을 호출한다.\"\"\"\n",
    "tests/test_smoke.py": "def test_import():\n    import {name}  # noqa: F401\n",
    "notebooks/README.md": "탐색·분석용. 재사용할 코드는 src/로 옮긴다.\n",
    "data/README.md": "데이터는 커밋하지 않는다. 내려받기·링크 방법을 여기에 쓴다.\n",
    "docs/README.md": "설계 문서·결정 기록.\n",
}


def scaffold(root: Path, name: str) -> list[Path]:
    created = []
    for rel, text in FILES.items():
        path = root / rel.format(name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text.format(name=name), encoding="utf-8")
            created.append(path)
    (root / "outputs").mkdir(exist_ok=True)
    return created


def tree(root: Path, prefix: str = "") -> list[str]:
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for i, p in enumerate(entries):
        last = i == len(entries) - 1
        lines.append(f"{prefix}{'└── ' if last else '├── '}{p.name}{'/' if p.is_dir() else ''}")
        if p.is_dir():
            lines += tree(p, prefix + ("    " if last else "│   "))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="연구 프로젝트 템플릿을 만든다.")
    parser.add_argument("root", type=Path, help="만들 프로젝트 디렉터리")
    parser.add_argument("--name", required=True, help="Python 패키지 이름 (snake_case)")
    args = parser.parse_args()
    if not args.name.isidentifier():
        parser.error(f"--name은 Python 식별자여야 한다: {args.name!r}")
    created = scaffold(args.root, args.name)
    print(f"{args.root.name}/  (새 파일 {len(created)}개)")
    print("\n".join(tree(args.root)))


if __name__ == "__main__":
    main()
