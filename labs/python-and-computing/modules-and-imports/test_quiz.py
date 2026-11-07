"""5.2 퀴즈(content/05-python-and-computing/02-modules-and-imports/quiz.toml) 정답을 실제 import·실행으로 검증한다."""
import subprocess
import sys
import tomllib
from pathlib import Path

QUIZ = Path(__file__).parents[3] / "content/05-python-and-computing/02-modules-and-imports/quiz.toml"
Q = {q["id"]: q for q in tomllib.loads(QUIZ.read_text(encoding="utf-8"))["questions"]}


def correct(qid):
    return [i for i, c in enumerate(Q[qid]["choices"]) if c.get("correct")]


def run(args, cwd):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)


def test_import_runs_once(tmp_path):
    (tmp_path / "consts.py").write_text('print("loaded")\n')
    (tmp_path / "model.py").write_text("import consts\n")
    (tmp_path / "data.py").write_text("import consts\n")
    (tmp_path / "train.py").write_text("import consts\nimport consts\nimport model\nimport data\n")
    out = run(["train.py"], tmp_path).stdout
    assert out.count("loaded") == Q["import-runs-once"]["answer"] == 1


def test_from_import_binding(tmp_path):
    (tmp_path / "config.py").write_text("LR = 1e-3\n")
    (tmp_path / "other.py").write_text("import config\nconfig.LR = 1e-2\n")
    (tmp_path / "train.py").write_text("from config import LR\nimport other\nprint(LR)\n")
    assert run(["train.py"], tmp_path).stdout.strip() == "0.001"
    assert correct("from-import-binding") == [1]


def test_script_sys_path(tmp_path):
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg/__init__.py").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").write_text("import sys, os\nprint(sys.path[0] == os.path.dirname(os.path.abspath(__file__)))\nimport mypkg\n")
    p = run(["scripts/train.py"], tmp_path)
    assert p.stdout.strip() == "True" and "ModuleNotFoundError" in p.stderr
    assert run(["-c", "import mypkg"], tmp_path).returncode == 0
    assert correct("script-sys-path") == [0]


def test_relative_import_as_script(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/util.py").write_text("X = 1\n")
    (tmp_path / "pkg/mod.py").write_text("from .util import X\nprint(X)\n")
    p = run(["pkg/mod.py"], tmp_path)
    assert "attempted relative import with no known parent package" in p.stderr
    assert run(["-m", "pkg.mod"], tmp_path).stdout.strip() == "1"
    # sys.path를 고쳐도 __package__가 비어 있어 여전히 실패한다
    (tmp_path / "pkg/mod2.py").write_text("import sys\nsys.path.append('..')\nfrom .util import X\n")
    assert "no known parent package" in run(["pkg/mod2.py"], tmp_path).stderr
    assert correct("relative-import-as-script") == [0]


def test_src_layout_not_importable_from_root(tmp_path):
    (tmp_path / "src/mypkg").mkdir(parents=True)
    (tmp_path / "src/mypkg/__init__.py").write_text("")
    assert run(["-c", "import mypkg"], tmp_path).returncode != 0
    assert correct("src-layout-benefits") == [0, 1]


def test_main_guard(tmp_path):
    (tmp_path / "analysis.py").write_text(
        "def load_results():\n    return 1\n"
        "def run_experiment():\n    print('experiment ran')\n"
        "run_experiment()\n"
    )
    (tmp_path / "user.py").write_text("from analysis import load_results\n")
    assert "experiment ran" in run(["user.py"], tmp_path).stdout
    assert correct("main-guard") == [1]
