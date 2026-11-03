#!/usr/bin/env python3
"""책 저장소 관리 도구 (표준 라이브러리만 사용).

사용법:
    python3 scripts/book.py check            # 디렉터리·파일명·frontmatter·선수 지식·링크 검증
    python3 scripts/book.py outline          # content/에서 OUTLINE.md 생성
    python3 scripts/book.py outline --check  # OUTLINE.md가 최신인지 확인(CI용)
    python3 scripts/book.py stats            # 상태·등급·트랙별 집계
    python3 scripts/book.py new-chapter 07-linear-algebra matrix-exponential "행렬 지수함수" "Matrix Exponential"

frontmatter는 YAML의 작은 부분집합만 허용한다(editorial/conventions.md 참고):
    key: 값 | key: "따옴표 문자열" | key: [a, b, c]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
OUTLINE = ROOT / "OUTLINE.md"
TEMPLATE = ROOT / "templates" / "chapter.md"

SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
PART_DIR = re.compile(rf"^(\d{{2}})-({SLUG})$")
CHAPTER_DIR = re.compile(rf"^(\d{{2}})-({SLUG})$")
APPENDIX_PART = "appendices"
APPENDIX_DIR = re.compile(rf"^([a-z])-({SLUG})$")
ASSET_FILE = re.compile(rf"^{SLUG}\.[a-z0-9]+$")
LINK = re.compile(r"\]\(([^)\s]+)\)")

PART_FIELDS = ("id", "title", "title_en", "track", "status", "summary")
CHAPTER_FIELDS = ("id", "title", "title_en", "tier", "status", "prerequisites")
REQUIRED_HEADINGS = ("학습 목표", "실습", "실패 사례", "통과 기준")


# --------------------------------------------------------------------------- model
@dataclass
class Chapter:
    path: Path
    label: str  # "7.2", "A"
    order: str
    slug: str
    meta: dict
    headings: list[str] = field(default_factory=list)
    body: str = ""


@dataclass
class Part:
    path: Path
    label: str  # "7", "부록"
    slug: str
    meta: dict
    chapters: list[Chapter] = field(default_factory=list)


class Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: Path, msg: str) -> None:
        self.items.append(f"{path.relative_to(ROOT)}: {msg}")


# ---------------------------------------------------------------------- parsing
def parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [x.strip() for x in raw[1:-1].split(",") if x.strip()]
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"')
    return raw


def read_doc(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("frontmatter(---)로 시작하지 않음")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("frontmatter가 닫히지 않음")
    meta = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep or not re.fullmatch(r"[a-z_]+", key):
            raise ValueError(f"해석할 수 없는 frontmatter 줄: {line!r}")
        meta[key] = parse_value(value)
    return meta, text[end + 5 :]


def load_config() -> dict:
    with open(ROOT / "book.toml", "rb") as f:
        return tomllib.load(f)


def load_book(errors: Errors) -> list[Part]:
    parts: list[Part] = []
    for pdir in sorted(p for p in CONTENT.iterdir() if p.is_dir()):
        if pdir.name == APPENDIX_PART:
            label, slug, chapter_re = "부록", APPENDIX_PART, APPENDIX_DIR
        elif m := PART_DIR.match(pdir.name):
            label, slug, chapter_re = str(int(m.group(1))), m.group(2), CHAPTER_DIR
        else:
            errors.add(pdir, "부 디렉터리 이름은 'NN-kebab-slug' 또는 'appendices'여야 함")
            continue
        index = pdir / "index.md"
        if not index.exists():
            errors.add(pdir, "index.md 없음")
            continue
        try:
            meta, _ = read_doc(index)
        except ValueError as e:
            errors.add(index, str(e))
            continue
        part = Part(pdir, label, slug, meta)
        for cdir in sorted(c for c in pdir.iterdir() if c.is_dir()):
            m = chapter_re.match(cdir.name)
            if not m:
                errors.add(cdir, f"장 디렉터리 이름 규칙 위반 ({chapter_re.pattern})")
                continue
            cindex = cdir / "index.md"
            if not cindex.exists():
                errors.add(cdir, "index.md 없음")
                continue
            try:
                cmeta, body = read_doc(cindex)
            except ValueError as e:
                errors.add(cindex, str(e))
                continue
            order = m.group(1)
            clabel = order.upper() if slug == APPENDIX_PART else f"{label}.{int(order)}"
            headings = re.findall(r"^## (.+?)\s*$", body, flags=re.M)
            part.chapters.append(Chapter(cindex, clabel, order, m.group(2), cmeta, headings, body))
        parts.append(part)
    return parts


# ----------------------------------------------------------------------- check
def check(parts: list[Part], config: dict, errors: Errors) -> None:
    tracks = {t["key"] for t in config["tracks"]}
    tiers, statuses = set(config["tiers"]), set(config["statuses"])
    ids: dict[str, Chapter] = {}

    for part in parts:
        index = part.path / "index.md"
        for f in PART_FIELDS:
            if f not in part.meta:
                errors.add(index, f"필수 필드 '{f}' 없음")
        if part.meta.get("id") != part.slug:
            errors.add(index, f"id는 디렉터리 slug '{part.slug}'와 같아야 함")
        if part.meta.get("track") not in tracks:
            errors.add(index, f"알 수 없는 track '{part.meta.get('track')}'")
        if part.meta.get("status") not in statuses:
            errors.add(index, f"알 수 없는 status '{part.meta.get('status')}'")

        if part.slug != APPENDIX_PART:
            expected = [f"{i:02d}" for i in range(1, len(part.chapters) + 1)]
            if [c.order for c in part.chapters] != expected:
                errors.add(part.path, "장 번호가 01부터 연속되지 않음")

        for ch in part.chapters:
            for f in CHAPTER_FIELDS:
                if f not in ch.meta:
                    errors.add(ch.path, f"필수 필드 '{f}' 없음")
            cid = f"{part.slug}/{ch.slug}"
            if ch.meta.get("id") != cid:
                errors.add(ch.path, f"id는 '{cid}'여야 함")
            if cid in ids:
                errors.add(ch.path, f"중복 id '{cid}'")
            ids[cid] = ch
            if ch.meta.get("tier") not in tiers:
                errors.add(ch.path, f"알 수 없는 tier '{ch.meta.get('tier')}'")
            if ch.meta.get("status") not in statuses:
                errors.add(ch.path, f"알 수 없는 status '{ch.meta.get('status')}'")
            if part.slug != APPENDIX_PART:
                for h in REQUIRED_HEADINGS:
                    if h not in ch.headings:
                        errors.add(ch.path, f"필수 절 '## {h}' 없음")
            for other in ch.path.parent.rglob("*"):
                if other.is_file() and other.name != "index.md" and not ASSET_FILE.match(other.name):
                    errors.add(other, "파일 이름은 소문자 kebab-case여야 함")
            for target in LINK.findall(ch.body):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                if not (ch.path.parent / target.split("#")[0]).resolve().exists():
                    errors.add(ch.path, f"깨진 상대 링크 '{target}'")

    # 선수 지식: 존재 여부와 순환
    graph: dict[str, list[str]] = {}
    for cid, ch in ids.items():
        prereqs = ch.meta.get("prerequisites") or []
        if not isinstance(prereqs, list):
            errors.add(ch.path, "prerequisites는 [a, b] 형식의 목록이어야 함")
            continue
        for p in prereqs:
            if p not in ids:
                errors.add(ch.path, f"존재하지 않는 선수 지식 '{p}'")
            elif p == cid:
                errors.add(ch.path, "자기 자신을 선수 지식으로 지정함")
        graph[cid] = [p for p in prereqs if p in ids]

    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1:
                cycle = " → ".join(stack[stack.index(nxt):] + [nxt]) if nxt in stack else nxt
                errors.add(ids[node].path, f"선수 지식 순환: {cycle}")
            elif state.get(nxt) is None:
                visit(nxt, stack + [nxt])
        state[node] = 2

    for node in graph:
        if node not in state:
            visit(node, [node])


# --------------------------------------------------------------------- outline
def render_outline(parts: list[Part], config: dict) -> str:
    tiers = config["tiers"]
    track_names = {t["key"]: t["name"] for t in config["tracks"]}
    titles = {f"{p.slug}/{c.slug}": (c.label, c.meta.get("title", "")) for p in parts for c in p.chapters}
    book = config["book"]
    n_ch = sum(len(p.chapters) for p in parts)

    out = [
        "<!-- 이 파일은 `python3 scripts/book.py outline`으로 생성된다. 직접 수정하지 말 것. -->",
        "",
        f"# {book['title']} — 전체 목차",
        "",
        f"> {book['subtitle']}",
        "",
        f"{len(parts)}개 부 · {n_ch}개 장. 각 장의 세부 bullet과 실습·실패 사례·통과 기준은 장 파일에 있다.",
        "등급: " + " · ".join(f"**{v}**({k})" for k, v in tiers.items()),
        "",
        "## 한눈에 보기",
        "",
        "| 부 | 제목 | 트랙 | 장 수 | 필수 |",
        "|---|---|---|---:|---:|",
    ]
    for p in parts:
        core = sum(c.meta.get("tier") == "core" for c in p.chapters)
        rel = (p.path / "index.md").relative_to(ROOT).as_posix()
        out.append(
            f"| {p.label} | [{p.meta.get('title')}]({rel}) | {track_names.get(p.meta.get('track'), '?')}"
            f" | {len(p.chapters)} | {core} |"
        )
    out.append("")

    current_track = None
    for p in parts:
        track = p.meta.get("track")
        if track != current_track:
            out += [f"## 트랙: {track_names.get(track, track)}", ""]
            current_track = track
        head = "부록" if p.slug == APPENDIX_PART else f"제{p.label}부"
        out += [f"### {head}. {p.meta.get('title')} <sub>{p.meta.get('title_en')}</sub>", ""]
        out += [f"_{p.meta.get('summary')}_", ""]
        for c in p.chapters:
            rel = c.path.relative_to(ROOT).as_posix()
            tier = tiers.get(c.meta.get("tier"), "?")
            out.append(f"- **{c.label}** [{c.meta.get('title')}]({rel}) · `{tier}`")
            sections = [h for h in c.headings if h not in REQUIRED_HEADINGS]
            if sections:
                out.append(f"  - {' · '.join(sections)}")
            prereqs = c.meta.get("prerequisites") or []
            if prereqs:
                refs = ", ".join(f"{titles[x][0]} {titles[x][1]}" for x in prereqs if x in titles)
                out.append(f"  - 선수: {refs}")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------- stats
def stats(parts: list[Part], config: dict) -> str:
    from collections import Counter

    chapters = [c for p in parts for c in p.chapters]
    lines = [f"부 {len(parts)} · 장 {len(chapters)}"]
    for key, names in (("status", config["statuses"]), ("tier", config["tiers"])):
        cnt = Counter(c.meta.get(key) for c in chapters)
        lines.append(f"{key}: " + ", ".join(f"{names.get(k, k)}({k}) {v}" for k, v in cnt.most_common()))
    track_cnt = Counter()
    for p in parts:
        track_cnt[p.meta.get("track")] += len(p.chapters)
    lines.append("track: " + ", ".join(f"{k} {v}" for k, v in track_cnt.items()))
    return "\n".join(lines)


# ------------------------------------------------------------------ new-chapter
def new_chapter(part_dir: str, slug: str, title: str, title_en: str) -> Path:
    pdir = CONTENT / part_dir
    if not pdir.is_dir():
        sys.exit(f"부 디렉터리 없음: {pdir}")
    if not re.fullmatch(SLUG, slug):
        sys.exit("slug는 소문자 kebab-case여야 함")
    if part_dir == APPENDIX_PART:
        sys.exit("부록 장은 직접 만든다 (a-, b-, ... 접두사)")
    m = PART_DIR.match(part_dir)
    if not m:
        sys.exit("부 디렉터리 이름은 'NN-slug' 형식이어야 함")
    part_slug = m.group(2)
    numbers = [int(c.name[:2]) for c in pdir.iterdir() if c.is_dir() and CHAPTER_DIR.match(c.name)]
    order = f"{max(numbers, default=0) + 1:02d}"
    cdir = pdir / f"{order}-{slug}"
    cdir.mkdir()
    text = TEMPLATE.read_text(encoding="utf-8")
    text = (
        text.replace("{{id}}", f"{part_slug}/{slug}")
        .replace("{{title}}", title)
        .replace("{{title_en}}", title_en)
    )
    (cdir / "index.md").write_text(text, encoding="utf-8")
    return cdir / "index.md"


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="저장소 규칙 검증")
    o = sub.add_parser("outline", help="OUTLINE.md 생성")
    o.add_argument("--check", action="store_true", help="파일을 쓰지 않고 최신 여부만 확인")
    sub.add_parser("stats", help="집계 출력")
    n = sub.add_parser("new-chapter", help="템플릿으로 새 장을 부의 끝에 추가")
    n.add_argument("part_dir", help="예: 07-linear-algebra")
    n.add_argument("slug", help="예: matrix-exponential")
    n.add_argument("title")
    n.add_argument("title_en")
    args = ap.parse_args()

    if args.cmd == "new-chapter":
        print(new_chapter(args.part_dir, args.slug, args.title, args.title_en).relative_to(ROOT))
        return 0

    config = load_config()
    errors = Errors()
    parts = load_book(errors)

    if args.cmd == "check":
        check(parts, config, errors)
        if not OUTLINE.exists() or OUTLINE.read_text(encoding="utf-8") != render_outline(parts, config):
            errors.items.append("OUTLINE.md: 최신이 아님 — `python3 scripts/book.py outline` 실행")
        for e in errors.items:
            print(f"error: {e}")
        print(f"{len(errors.items)} error(s)" if errors.items else "ok")
        return 1 if errors.items else 0
    if args.cmd == "outline":
        text = render_outline(parts, config)
        if args.check:
            ok = OUTLINE.exists() and OUTLINE.read_text(encoding="utf-8") == text
            print("ok" if ok else "OUTLINE.md가 최신이 아님")
            return 0 if ok else 1
        OUTLINE.write_text(text, encoding="utf-8")
        print(f"wrote {OUTLINE.relative_to(ROOT)}")
        return 0
    if args.cmd == "stats":
        print(stats(parts, config))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
