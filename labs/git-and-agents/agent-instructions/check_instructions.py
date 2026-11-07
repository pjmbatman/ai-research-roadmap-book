"""4.4 실습: 에이전트 지침 파일을 기계적으로 점검한다.

    python check_instructions.py <프로젝트 디렉터리>

1. AGENTS.md의 ```bash 블록에 적힌 명령을 그 디렉터리에서 실제로 실행해, 지침과 실제가 어긋났는지 본다.
2. skills/*/SKILL.md의 frontmatter(name·description)를 검사한다.
3. AGENTS.md가 가리키는 상대 경로가 실제로 있는지 본다.
점검을 통과해도 지침이 '좋은' 것은 아니다. 처음 보는 에이전트에게 실제로 맡겨 보는 것이 최종 시험이다.
"""
import re
import subprocess
import sys
from pathlib import Path

BASH_BLOCK = re.compile(r"^[ \t]*```bash\n(.*?)^[ \t]*```", re.M | re.S)
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
PATH_REF = re.compile(r"`([\w./-]+/[\w./-]+)`")


def extract_commands(md_text):
    """```bash 블록의 명령 줄(빈 줄·주석 제외)을 순서대로 반환한다."""
    cmds = []
    for block in BASH_BLOCK.findall(md_text):
        cmds += [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return cmds


def parse_frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    return dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)


def check_skill(path):
    """SKILL.md 하나의 문제 목록을 반환한다(없으면 빈 리스트)."""
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    if fm is None:
        return [f"{path}: frontmatter(--- ... ---)가 없음"]
    name, desc = fm.get("name", "").strip(), fm.get("description", "").strip()
    problems = []
    if not NAME.match(name):
        problems.append(f"{path}: name '{name}'은 소문자·숫자·하이픈이어야 함")
    if name != path.parent.name:
        problems.append(f"{path}: name '{name}'이 디렉터리 이름 '{path.parent.name}'과 다름")
    if len(desc) < 40:
        problems.append(f"{path}: description이 너무 짧음 — 무엇을 하는지와 언제 쓰는지를 함께 쓴다")
    return problems


def run_commands(cmds, cwd, timeout_s=60):
    """명령을 차례로 실행하고 (명령, 종료 코드) 목록을 반환한다."""
    results = []
    for cmd in cmds:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        results.append((cmd, r.returncode))
    return results


def check_project(root):
    root = Path(root)
    agents = root / "AGENTS.md"
    if not agents.is_file():
        return [f"{root}: AGENTS.md가 없음"]
    text = agents.read_text(encoding="utf-8")
    problems = []
    for ref in PATH_REF.findall(text):
        if not (root / ref).exists():
            problems.append(f"AGENTS.md가 가리키는 경로가 없음: {ref}")
    for skill in sorted(root.glob("skills/*/SKILL.md")):
        problems += check_skill(skill)
        for cmd, code in run_commands(extract_commands(skill.read_text(encoding="utf-8")), root):
            if code != 0:
                problems.append(f"{skill.relative_to(root)}의 명령 실패(종료 코드 {code}): {cmd}")
    cmds = extract_commands(text)
    if not cmds:
        problems.append("AGENTS.md에 ```bash 블록으로 적은 실행·테스트 명령이 없음")
    for cmd, code in run_commands(cmds, root):
        if code != 0:
            problems.append(f"AGENTS.md의 명령 실패(종료 코드 {code}): {cmd}")
    return problems


if __name__ == "__main__":
    found = check_project(sys.argv[1] if len(sys.argv) > 1 else ".")
    for p in found:
        print("문제:", p)
    print("통과" if not found else f"문제 {len(found)}개")
    sys.exit(1 if found else 0)
