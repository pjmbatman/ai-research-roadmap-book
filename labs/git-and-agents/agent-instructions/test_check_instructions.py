"""4.4 실습: 예시 프로젝트의 지침이 점검을 통과하고, 흔한 지침 결함은 점검에 걸리는지 확인한다."""
import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("check_instructions", HERE / "check_instructions.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)

SAMPLE = HERE / "sample_project"


def copy_sample(tmp_path):
    dst = tmp_path / "proj"
    shutil.copytree(SAMPLE, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dst


def test_sample_project_passes():
    assert ci.check_project(SAMPLE) == []


def test_extract_commands_skips_comments():
    md = "x\n```bash\n# 주석\npython a.py\n\npytest -q\n```\n"
    assert ci.extract_commands(md) == ["python a.py", "pytest -q"]


def test_stale_command_is_caught(tmp_path):
    proj = copy_sample(tmp_path)
    agents = proj / "AGENTS.md"
    agents.write_text(agents.read_text(encoding="utf-8").replace("src/halflife.py", "scripts/run.py"), encoding="utf-8")
    problems = ci.check_project(proj)
    assert any("scripts/run.py" in p and "실패" in p for p in problems)


def test_dangling_path_is_caught(tmp_path):
    proj = copy_sample(tmp_path)
    shutil.rmtree(proj / "skills")
    problems = ci.check_project(proj)
    assert any("skills/add-decay-dataset/SKILL.md" in p for p in problems)


def test_bad_skill_frontmatter(tmp_path):
    d = tmp_path / "skills" / "my-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: My_Skill\ndescription: 테스트\n---\n# x\n", encoding="utf-8")
    problems = ci.check_skill(d / "SKILL.md")
    assert len(problems) == 3                                  # name 형식, 디렉터리 불일치, 짧은 description


def test_missing_frontmatter(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("# 제목만 있음\n", encoding="utf-8")
    assert "frontmatter" in ci.check_skill(p)[0]
