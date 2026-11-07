"""4.2 퀴즈(content/04-git-and-agents/02-github-collaboration/quiz.toml)와 research.gitignore를 실제 git 저장소에서 검증한다.

각 테스트는 임시 디렉터리에 새 저장소를 만든다. git이 없으면 건너뛴다.
"""
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).parent
QUIZ = HERE.parents[2] / "content/04-git-and-agents/02-github-collaboration/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없음")


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def make_git(cwd):
    def git(*args, check=True):
        cmd = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
               "-c", "init.defaultBranch=main", *args]
        r = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        return r if not check else r.stdout.strip()
    return git


@pytest.fixture
def repo(tmp_path):
    git = make_git(tmp_path)
    git("init", "-q")
    git.path = tmp_path
    return git


def write(root, rel, text="x\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_research_gitignore(repo):
    shutil.copy(HERE / "research.gitignore", repo.path / ".gitignore")
    ignored = ["data/raw.csv", "outputs/fig1.png", "model.ckpt", "weights.pt", ".env", ".env.local",
               ".venv/bin/python", "src/__pycache__/m.cpython-311.pyc"]
    kept = ["data/README.md", ".env.example", "src/train.py", "configs/train.yaml", "scripts/download_data.sh"]
    for rel in ignored + kept:
        write(repo.path, rel)
    for rel in ignored:
        assert repo("check-ignore", "-q", rel, check=False).returncode == 0, f"{rel}은 무시되어야 함"
    for rel in kept:
        assert repo("check-ignore", "-q", rel, check=False).returncode == 1, f"{rel}은 추적되어야 함"


def test_fetch_vs_pull(tmp_path):
    make_git(tmp_path)("init", "-q", "--bare", "remote.git")
    a, b = tmp_path / "a", tmp_path / "b"
    make_git(tmp_path)("clone", "-q", "remote.git", "a")
    ga = make_git(a)
    write(a, "f.txt", "1\n"); ga("add", "f.txt"); ga("commit", "-q", "-m", "1"); ga("push", "-q", "origin", "main")
    make_git(tmp_path)("clone", "-q", "remote.git", "b")
    gb = make_git(b)
    write(a, "f.txt", "2\n"); ga("commit", "-q", "-am", "2"); ga("push", "-q", "origin", "main")
    before = gb("rev-parse", "main")
    gb("fetch", "-q", "origin")
    assert gb("rev-parse", "main") == before                         # 로컬 branch 그대로
    assert gb("rev-parse", "origin/main") == ga("rev-parse", "main")  # 원격 추적 branch만 갱신
    assert (b / "f.txt").read_text() == "1\n"                         # working tree 그대로
    assert correct("fetch-vs-pull") == [0]

    # b가 오래된 main에서 commit하고 push하면 거부된다
    write(b, "g.txt", "b\n"); gb("add", "g.txt"); gb("commit", "-q", "-m", "b")
    r = make_git(b)("-c", "advice.pushFetchFirst=false", "push", "origin", "main", check=False)
    assert r.returncode != 0 and "rejected" in r.stderr
    gb("pull", "-q", "--no-rebase", "origin", "main")                # 받아서 합친 뒤에는
    gb("push", "-q", "origin", "main")                               # push가 된다
    assert correct("push-rejected") == [0]


def test_gitignore_tracked_file(repo):
    write(repo.path, "data/raw.csv", "a\n")
    repo("add", "data/raw.csv"); repo("commit", "-q", "-m", "data")
    write(repo.path, ".gitignore", "data/\n")
    write(repo.path, "data/raw.csv", "b\n")
    assert "data/raw.csv" in repo("status", "--porcelain")            # 무시되지 않는다
    repo("rm", "-q", "--cached", "data/raw.csv")
    repo("add", ".gitignore"); repo("commit", "-q", "-m", "untrack")
    assert "data/raw.csv" not in repo("status", "--porcelain")
    assert (repo.path / "data/raw.csv").exists()                     # 디스크에는 남는다
    assert repo("show", "HEAD~1:data/raw.csv") == "a"                # 과거 commit에도 남는다
    assert correct("gitignore-tracked-file") == [0]


def test_checkpoint_in_history(repo):
    write(repo.path, "model.ckpt", "W" * 1000)
    repo("add", "model.ckpt"); repo("commit", "-q", "-m", "ckpt")
    blob = repo("rev-parse", "HEAD:model.ckpt")
    repo("rm", "-q", "model.ckpt"); repo("commit", "-q", "-m", "remove")
    assert repo("cat-file", "-t", blob) == "blob"                    # 여전히 저장소 안에 있다
    clone = repo.path.parent / "clone"
    make_git(repo.path.parent)("clone", "-q", str(repo.path), "clone")
    assert make_git(clone)("cat-file", "-t", blob) == "blob"         # clone도 받아 간다
    assert correct("checkpoint-in-history") == [0, 2]
