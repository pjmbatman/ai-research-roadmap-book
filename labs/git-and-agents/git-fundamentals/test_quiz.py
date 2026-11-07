"""4.1 퀴즈(content/04-git-and-agents/01-git-fundamentals/quiz.toml) 정답을 실제 git 저장소에서 검증한다.

각 테스트는 임시 디렉터리에 새 저장소를 만든다. git이 없으면 건너뛴다.
"""
import hashlib
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

QUIZ = Path(__file__).parents[3] / "content/04-git-and-agents/01-git-fundamentals/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git이 없음")


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        cmd = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
               "-c", "init.defaultBranch=main", *args]
        return subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    def commit(name, text, msg):
        (tmp_path / name).write_text(text)
        git("add", name)
        git("commit", "-q", "-m", msg)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    git.commit, git.path = commit, tmp_path
    return git


def test_blob_id_matches_cell():
    content = b"hello\n"
    assert hashlib.sha1(b"blob 6\0" + content).hexdigest() == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_staged_version_committed(repo):
    repo.commit("train.py", "lr = 1e-1\n", "init")
    (repo.path / "train.py").write_text("lr = 1e-3\n")
    repo("add", "train.py")
    (repo.path / "train.py").write_text("lr = 3e-4\n")
    repo("commit", "-q", "-m", "lr 변경")
    assert repo("show", "HEAD:train.py") == "lr = 1e-3"
    assert correct("staged-version-committed") == [0]


def test_history_is_content_addressed(repo):
    a = repo.commit("train.py", "x\n", "A")
    b = repo.commit("config.yaml", "lerning_rate: 1\n", "B")
    c = repo.commit("c.txt", "c\n", "C")
    d = repo.commit("d.txt", "d\n", "D")
    train_blob = repo("rev-parse", f"{b}:train.py")
    # B를 고쳐 이력을 다시 쓴다: B를 고친 commit 위에 C, D를 다시 적용
    repo("switch", "-q", "--detach", b)
    (repo.path / "config.yaml").write_text("learning_rate: 1\n")
    repo("commit", "-q", "--amend", "-a", "-m", "B")
    repo("cherry-pick", c, d)
    new = repo("rev-list", "--reverse", "HEAD").split()
    assert new[0] == a                                     # A 그대로
    assert new[1] != b and new[2] != c and new[3] != d     # B, C, D 모두 바뀜
    assert repo("rev-parse", "HEAD~2:train.py") == train_blob
    assert correct("history-is-content-addressed") == [0, 1]


def test_fast_forward_merge(repo):
    base = repo.commit("a.txt", "a\n", "base")
    repo("switch", "-q", "-c", "feature")
    tip = repo.commit("b.txt", "b\n", "feat")
    repo("switch", "-q", "main")
    repo("merge", "-q", "feature")
    assert repo("rev-parse", "main") == tip
    assert repo("rev-list", "--count", "HEAD") == "2" and base != tip
    assert correct("fast-forward-merge") == [0]


def test_rebase_new_commits(repo):
    repo.commit("base.txt", "0\n", "base")
    repo("switch", "-q", "-c", "feature")
    old = [repo.commit(f"f{i}.txt", f"{i}\n", f"f{i}") for i in range(3)]
    repo("switch", "-q", "main")
    main_new = [repo.commit(f"m{i}.txt", f"{i}\n", f"m{i}") for i in range(2)]
    repo("switch", "-q", "feature")
    repo("rebase", "-q", "main")
    after = set(repo("rev-list", "HEAD").split())
    created = after - set(old) - set(main_new) - {repo("rev-parse", "main~2")}
    assert len(created) == Q["rebase-new-commits"]["answer"] == 3
    assert not (set(old) & after)                          # 원래 commit은 더 이상 feature에 없다


def test_rebase_vs_merge():
    assert correct("rebase-vs-merge") == [0]


def test_recover_after_reset_hard(repo):
    repo.commit("a.txt", "a\n", "a")
    lost = [repo.commit(f"{n}.txt", n, n) for n in ("train", "eval")]
    repo("reset", "-q", "--hard", "HEAD~2")
    assert lost[-1] not in repo("rev-list", "HEAD")
    assert repo("rev-parse", "main@{1}") == lost[-1]       # reflog에 남아 있다
    repo("reset", "-q", "--hard", "main@{1}")
    assert repo("rev-parse", "HEAD") == lost[-1]
    assert correct("recover-after-reset-hard") == [0]


def test_undo_on_shared_branch(repo):
    repo.commit("metric.py", "ok\n", "a")
    bad = repo.commit("metric.py", "broken\n", "bad")
    repo("revert", "--no-edit", bad)
    assert repo("show", "HEAD:metric.py") == "ok"
    assert bad in repo("rev-list", "HEAD")                 # 문제 commit은 이력에 남는다
    assert correct("undo-on-shared-branch") == [0]
