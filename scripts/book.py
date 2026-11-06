#!/usr/bin/env python3
"""책 저장소 관리 도구 (표준 라이브러리만 사용, Python 3.11+).

사용법:
    python3 scripts/book.py check            # 구조·frontmatter·로드맵 규칙·퀴즈·링크·OUTLINE 최신 여부 검증
    python3 scripts/book.py outline          # content/에서 OUTLINE.md 생성
    python3 scripts/book.py outline --check  # OUTLINE.md가 최신인지 확인
    python3 scripts/book.py stats            # 상태·등급·트랙·퀴즈 집계
    python3 scripts/book.py new-chapter 07-linear-algebra matrix-exponential "행렬 지수함수" "Matrix Exponential"

frontmatter는 YAML의 작은 부분집합만 허용한다(editorial/conventions.md 참고):
    key: 값 | key: "따옴표 문자열" | key: [a, b, c]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import Counter, defaultdict
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
TRACK_CODE = re.compile(r"^[A-Z]{3}$")

PART_FIELDS = ("id", "title", "title_en", "track", "status", "summary")
CHAPTER_FIELDS = ("id", "title", "title_en", "tier", "status", "prerequisites")
TRACK_FIELDS = ("key", "code", "name", "name_en", "layer", "question", "outcome", "assessment", "capstone")
REQUIRED_HEADINGS = ("학습 목표", "실습", "실패 사례", "통과 기준")
QUIZ_FILE = "quiz.toml"
QUIZ_TYPES = ("single", "multiple", "numeric")
WIDGETS = ROOT / "data" / "widgets.toml"
LABS = ROOT / "labs"
WIDGET_STATUSES = ("planned", "prototype", "implemented")
WIDGET_KINDS = ("manipulate", "stepper", "simulate", "compare")
EXEC_TIERS = ("browser", "python", "gpu")
DIRECTIVE = re.compile(r'^::(widget|quiz|exercise)\{(.*)\}\s*$', flags=re.M)
DIRECTIVE_ATTR = re.compile(r'(\w+)="([^"]*)"')
FENCE_EXEC = re.compile(r'^```\w+\s*\{[^}\n]*\bexec="([^"]*)"', flags=re.M)
FENCE_EXEC_BODY = re.compile(r'^```\w+\s*\{[^}\n]*\bexec="[^"]*"[^\n]*\n(.*?)^```', flags=re.M | re.S)
CELL_SPLIT = re.compile(r"^# %% ", flags=re.M)


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
    quiz: list[dict] | None = None

    @property
    def objectives(self) -> int:
        m = re.search(r"^## 학습 목표\s*\n(.*?)(?=^## |\Z)", self.body, flags=re.M | re.S)
        return len(re.findall(r"^- \S", m.group(1), flags=re.M)) if m else 0


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

    def add(self, path: Path | str, msg: str) -> None:
        where = path.relative_to(ROOT) if isinstance(path, Path) else path
        self.items.append(f"{where}: {msg}")


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
            ch = Chapter(cindex, clabel, order, m.group(2), cmeta, headings, body)
            qpath = cdir / QUIZ_FILE
            if qpath.exists():
                try:
                    with open(qpath, "rb") as f:
                        ch.quiz = tomllib.load(f).get("questions", [])
                except tomllib.TOMLDecodeError as e:
                    errors.add(qpath, f"TOML 오류: {e}")
                    ch.quiz = []
            part.chapters.append(ch)
        parts.append(part)
    return parts


# ----------------------------------------------------------------------- check
def check_config(config: dict, errors: Errors) -> None:
    levels = {layer["level"] for layer in config.get("layers", [])}
    assessments = set(config.get("assessments", {}))
    keys, codes, last_layer = set(), set(), -1
    for t in config.get("tracks", []):
        where = f"book.toml [tracks.{t.get('key', '?')}]"
        for f in TRACK_FIELDS:
            if f not in t:
                errors.add(where, f"필수 필드 '{f}' 없음")
        if not re.fullmatch(SLUG, t.get("key", "")):
            errors.add(where, "key는 kebab-case여야 함")
        if not TRACK_CODE.match(t.get("code", "")):
            errors.add(where, "code는 영문 대문자 3자여야 함")
        if t.get("key") in keys or t.get("code") in codes:
            errors.add(where, "key 또는 code 중복")
        keys.add(t.get("key"))
        codes.add(t.get("code"))
        if t.get("layer") not in levels:
            errors.add(where, f"정의되지 않은 layer {t.get('layer')}")
        elif t["layer"] < last_layer:
            errors.add(where, "트랙은 layer 오름차순으로 나열해야 함")
        else:
            last_layer = t["layer"]
        for a in t.get("assessment", []):
            if a not in assessments:
                errors.add(where, f"정의되지 않은 평가 방식 '{a}'")


def check_quiz(ch: Chapter, assessments: set[str], errors: Errors) -> None:
    qpath = ch.path.parent / QUIZ_FILE
    if not ch.quiz:
        errors.add(qpath, "[[questions]]가 하나도 없음")
        return
    seen = set()
    for i, q in enumerate(ch.quiz, 1):
        where = f"{qpath.relative_to(ROOT)} #{i}"
        qid = q.get("id", "")
        if not re.fullmatch(SLUG, qid):
            errors.add(where, "id는 kebab-case여야 함")
        if qid in seen:
            errors.add(where, f"중복 id '{qid}'")
        seen.add(qid)
        for f in ("prompt", "explanation"):
            if not str(q.get(f, "")).strip():
                errors.add(where, f"'{f}'가 비어 있음")
        obj = q.get("objective")
        if not isinstance(obj, int) or not 1 <= obj <= max(ch.objectives, 1):
            errors.add(where, f"objective는 1–{ch.objectives} 사이의 학습 목표 번호여야 함")
        qtype = q.get("type")
        if qtype not in QUIZ_TYPES:
            errors.add(where, f"type은 {QUIZ_TYPES} 중 하나여야 함")
            continue
        if qtype == "numeric":
            if not isinstance(q.get("answer"), (int, float)):
                errors.add(where, "numeric 문항에는 수치 answer가 필요함")
            if not isinstance(q.get("tolerance", 0), (int, float)) or q.get("tolerance", 0) < 0:
                errors.add(where, "tolerance는 0 이상의 수여야 함")
            slider = q.get("slider")
            if slider is not None:
                lo, hi, step = slider.get("min"), slider.get("max"), slider.get("step")
                if not all(isinstance(v, (int, float)) for v in (lo, hi, step)) or step <= 0 or lo >= hi:
                    errors.add(where, "slider에는 min < max, step > 0인 수치가 필요함")
                elif isinstance(q.get("answer"), (int, float)):
                    ans = q["answer"]
                    if not lo <= ans <= hi:
                        errors.add(where, "정답이 slider 범위 밖에 있음")
                    elif abs((ans - lo) / step - round((ans - lo) / step)) > 1e-9 and q.get("tolerance", 0) < step / 2:
                        errors.add(where, "slider 눈금으로 정답에 도달할 수 없음 (step 또는 tolerance 조정)")
            continue
        choices = q.get("choices", [])
        if len(choices) < 2:
            errors.add(where, "선택지는 2개 이상이어야 함")
        if any(not str(c.get("text", "")).strip() for c in choices):
            errors.add(where, "비어 있는 선택지가 있음")
        n_correct = sum(bool(c.get("correct")) for c in choices)
        if qtype == "single" and n_correct != 1:
            errors.add(where, f"single 문항은 정답이 정확히 1개여야 함 (현재 {n_correct})")
        if qtype == "multiple" and not 1 <= n_correct < len(choices):
            errors.add(where, "multiple 문항은 정답이 1개 이상, 전체 미만이어야 함")


def load_widgets(errors: Errors) -> list[dict]:
    if not WIDGETS.exists():
        return []
    try:
        with open(WIDGETS, "rb") as f:
            widgets = tomllib.load(f).get("widgets", [])
    except tomllib.TOMLDecodeError as e:
        errors.add(WIDGETS, f"TOML 오류: {e}")
        return []
    seen = set()
    for w in widgets:
        name = w.get("name", "")
        if not re.fullmatch(SLUG, name) or name in seen:
            errors.add(WIDGETS, f"위젯 이름 '{name}'이 kebab-case가 아니거나 중복됨")
        seen.add(name)
        if w.get("status") not in WIDGET_STATUSES:
            errors.add(WIDGETS, f"위젯 '{name}': status는 {WIDGET_STATUSES} 중 하나")
        if w.get("kind") not in WIDGET_KINDS:
            errors.add(WIDGETS, f"위젯 '{name}': kind는 {WIDGET_KINDS} 중 하나")
        for f in ("title", "chapters", "interaction"):
            if not w.get(f):
                errors.add(WIDGETS, f"위젯 '{name}': '{f}' 없음")
    return widgets


def check_embeds(ch: Chapter, part_slug: str, widget_names: set[str], errors: Errors) -> None:
    """본문에 삽입된 위젯·퀴즈·연습문제·실행 코드 블록의 참조를 검증한다."""
    quiz_ids = {q.get("id") for q in ch.quiz or []}
    for kind, raw in DIRECTIVE.findall(ch.body):
        attrs = dict(DIRECTIVE_ATTR.findall(raw))
        if kind == "widget" and attrs.get("name") not in widget_names:
            errors.add(ch.path, f"data/widgets.toml에 없는 위젯 '{attrs.get('name')}'")
        elif kind == "quiz" and attrs.get("id") not in quiz_ids:
            errors.add(ch.path, f"{QUIZ_FILE}에 없는 문항 '{attrs.get('id')}'")
        elif kind == "exercise":
            ex = LABS / part_slug / ch.slug / "exercises" / attrs.get("id", "")
            if not attrs.get("id") or not ex.is_dir():
                errors.add(ch.path, f"연습문제 디렉터리 없음: {ex.relative_to(ROOT)}")
    for tier in FENCE_EXEC.findall(ch.body):
        if tier not in EXEC_TIERS:
            errors.add(ch.path, f"코드 블록 exec는 {EXEC_TIERS} 중 하나여야 함 (현재 '{tier}')")
    # 본문 실행 코드 블록은 labs/<부>/<장>/cells.py의 셀과 순서대로 같아야 한다(conventions §8).
    cells_py = LABS / part_slug / ch.slug / "cells.py"
    blocks = FENCE_EXEC_BODY.findall(ch.body)
    if cells_py.is_file():
        cells = [c.split("\n", 1)[1] if "\n" in c else "" for c in CELL_SPLIT.split(cells_py.read_text(encoding="utf-8"))[1:]]
        if len(blocks) != len(cells):
            errors.add(ch.path, f"실행 코드 블록 {len(blocks)}개, {cells_py.relative_to(ROOT)} 셀 {len(cells)}개")
        for i, (block, cell) in enumerate(zip(blocks, cells), 1):
            if block.strip() != cell.strip():
                errors.add(ch.path, f"실행 코드 블록 {i}이 {cells_py.relative_to(ROOT)}의 cell {i}과 다름")
    elif blocks:
        errors.add(ch.path, f"실행 코드 블록이 있는데 {cells_py.relative_to(ROOT)}가 없음")


def check(parts: list[Part], config: dict, errors: Errors) -> None:
    check_config(config, errors)
    tracks = {t["key"]: t for t in config["tracks"]}
    track_index = {t["key"]: i for i, t in enumerate(config["tracks"])}
    tiers, statuses = set(config["tiers"]), set(config["statuses"])
    assessments = set(config["assessments"])
    ids: dict[str, Chapter] = {}
    chapter_track: dict[str, str] = {}
    last_index = -1
    widgets = load_widgets(errors)
    widget_names = {w.get("name") for w in widgets}

    for part in parts:
        index = part.path / "index.md"
        for f in PART_FIELDS:
            if f not in part.meta:
                errors.add(index, f"필수 필드 '{f}' 없음")
        if part.meta.get("id") != part.slug:
            errors.add(index, f"id는 디렉터리 slug '{part.slug}'와 같아야 함")
        track = part.meta.get("track")
        if track not in tracks:
            errors.add(index, f"알 수 없는 track '{track}'")
        elif track_index[track] < last_index:
            errors.add(index, "부의 순서가 book.toml의 트랙 순서(레이어 순서)를 따르지 않음")
        else:
            last_index = track_index[track]
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
            chapter_track[cid] = track
            if ch.meta.get("tier") not in tiers:
                errors.add(ch.path, f"알 수 없는 tier '{ch.meta.get('tier')}'")
            if ch.meta.get("status") not in statuses:
                errors.add(ch.path, f"알 수 없는 status '{ch.meta.get('status')}'")
            for a in ch.meta.get("assessment", []) or []:
                if a not in assessments:
                    errors.add(ch.path, f"정의되지 않은 평가 방식 '{a}'")
            if part.slug != APPENDIX_PART:
                for h in REQUIRED_HEADINGS:
                    if h not in ch.headings:
                        errors.add(ch.path, f"필수 절 '## {h}' 없음")
            if ch.quiz is not None:
                check_quiz(ch, assessments, errors)
            for other in ch.path.parent.rglob("*"):
                if other.is_file() and other.name != "index.md" and not ASSET_FILE.match(other.name):
                    errors.add(other, "파일 이름은 소문자 kebab-case여야 함")
            check_embeds(ch, part.slug, widget_names, errors)
            for target in LINK.findall(ch.body):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                if not (ch.path.parent / target.split("#")[0]).resolve().exists():
                    errors.add(ch.path, f"깨진 상대 링크 '{target}'")

    # 선수 지식: 존재 · 자기참조 · 레이어 방향
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
            else:
                mine, theirs = chapter_track[cid], chapter_track[p]
                if mine in tracks and theirs in tracks and tracks[theirs]["layer"] > tracks[mine]["layer"]:
                    errors.add(
                        ch.path,
                        f"레이어 역방향 선수 지식 '{p}' ({tracks[mine]['code']} L{tracks[mine]['layer']}"
                        f" → {tracks[theirs]['code']} L{tracks[theirs]['layer']})",
                    )
        graph[cid] = [p for p in prereqs if p in ids and p != cid]

    for w in widgets:
        for c in w.get("chapters", []):
            if c not in ids:
                errors.add(WIDGETS, f"위젯 '{w.get('name')}': 존재하지 않는 장 '{c}'")

    find_cycles(graph, lambda n: ids[n].path, "선수 지식 순환", errors)
    # 트랙 수준에서도 순환이 없어야 로드맵이 한 방향으로 읽힌다
    tgraph: dict[str, set[str]] = defaultdict(set)
    for cid, prereqs in graph.items():
        for p in prereqs:
            if chapter_track[cid] != chapter_track[p]:
                tgraph[chapter_track[cid]].add(chapter_track[p])
    find_cycles({k: sorted(v) for k, v in tgraph.items()}, lambda n: "book.toml", "트랙 의존 순환", errors)


def find_cycles(graph: dict[str, list[str]], where, label: str, errors: Errors) -> None:
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1:
                cycle = stack[stack.index(nxt):] + [nxt] if nxt in stack else [node, nxt]
                errors.add(where(node), f"{label}: {' → '.join(cycle)}")
            elif state.get(nxt) is None:
                visit(nxt, stack + [nxt])
        state[node] = 2

    for node in list(graph):
        if node not in state:
            visit(node, [node])


# --------------------------------------------------------------------- outline
def track_dependencies(parts: list[Part]) -> dict[str, Counter]:
    """트랙 → (선수 트랙 → 장 단위 선수 지식 수)."""
    track_of = {f"{p.slug}/{c.slug}": p.meta.get("track") for p in parts for c in p.chapters}
    deps: dict[str, Counter] = defaultdict(Counter)
    for p in parts:
        for c in p.chapters:
            for q in c.meta.get("prerequisites") or []:
                if q in track_of and track_of[q] != p.meta.get("track"):
                    deps[p.meta.get("track")][track_of[q]] += 1
    return deps


def render_outline(parts: list[Part], config: dict) -> str:
    tiers = config["tiers"]
    tracks = {t["key"]: t for t in config["tracks"]}
    layers = {layer["level"]: layer for layer in config["layers"]}
    assess = config["assessments"]
    titles = {f"{p.slug}/{c.slug}": (c.label, c.meta.get("title", "")) for p in parts for c in p.chapters}
    book = config["book"]
    n_ch = sum(len(p.chapters) for p in parts)
    n_quiz = sum(len(c.quiz or []) for p in parts for c in p.chapters)
    parts_by_track: dict[str, list[Part]] = defaultdict(list)
    for p in parts:
        parts_by_track[p.meta.get("track")].append(p)
    widgets_of: dict[str, list[str]] = defaultdict(list)
    for w in load_widgets(Errors()):
        for c in w.get("chapters", []):
            widgets_of[c].append(w.get("title", w.get("name")))
    n_widgets = len({w for ws in widgets_of.values() for w in ws})

    out = [
        "<!-- 이 파일은 `python3 scripts/book.py outline`으로 생성된다. 직접 수정하지 말 것. -->",
        "",
        f"# {book['title']} — 전체 목차",
        "",
        f"> {book['subtitle']}",
        "",
        f"{len(tracks)}개 트랙 · {len(parts)}개 부 · {n_ch}개 장 · 퀴즈 {n_quiz}문항 · 위젯 {n_widgets}개. "
        "각 장의 세부 bullet과 실습·실패 사례·통과 기준은 장 파일에 있다. "
        "트랙·레이어 설계 근거는 [editorial/roadmap-design.md](editorial/roadmap-design.md).",
        "",
        "등급: " + " · ".join(f"**{v}**({k})" for k, v in tiers.items()),
        "",
        "## 트랙 구성",
        "",
        "| 레이어 | 코드 | 트랙 | 핵심 질문 | 부 | 장 | 평가 |",
        "|---|---|---|---|---|---:|---|",
    ]
    for t in config["tracks"]:
        tparts = parts_by_track.get(t["key"], [])
        labels = ", ".join(p.label for p in tparts)
        n = sum(len(p.chapters) for p in tparts)
        layer = layers[t["layer"]]
        modes = " · ".join(assess[a] for a in t["assessment"]) or "—"
        out.append(
            f"| L{t['layer']} {layer['name']} | `{t['code']}` | **{t['name']}**<br>{t['name_en']}"
            f" | {t['question']} | {labels} | {n} | {modes} |"
        )
    out += [
        "",
        "## 트랙 간 의존",
        "",
        "장 단위 선수 지식(`prerequisites`)을 트랙 단위로 집계한 것이다. 숫자는 선수 지식 연결 수. "
        "모든 화살표는 같은 레이어이거나 아래 레이어를 향한다(`check`가 검증).",
        "",
        "| 트랙 | 기대는 트랙 |",
        "|---|---|",
    ]
    deps = track_dependencies(parts)
    for t in config["tracks"]:
        d = deps.get(t["key"])
        if not d:
            continue
        items = ", ".join(f"`{tracks[k]['code']}` {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
        out.append(f"| `{t['code']}` {t['name']} | {items} |")
    out.append("")

    current_track = None
    for p in parts:
        tkey = p.meta.get("track")
        if tkey != current_track:
            t = tracks.get(tkey, {})
            out += [
                f"## `{t.get('code', '?')}` {t.get('name', tkey)} <sub>{t.get('name_en', '')}</sub>",
                "",
                f"- **핵심 질문:** {t.get('question', '')}",
                f"- **도달 목표:** {t.get('outcome', '')}",
            ]
            if t.get("capstone"):
                out.append(f"- **트랙 졸업 과제:** {t['capstone']}")
            out.append("")
            current_track = tkey
        head = "부록" if p.slug == APPENDIX_PART else f"제{p.label}부"
        out += [f"### {head}. {p.meta.get('title')} <sub>{p.meta.get('title_en')}</sub>", ""]
        out += [f"_{p.meta.get('summary')}_", ""]
        for c in p.chapters:
            rel = c.path.relative_to(ROOT).as_posix()
            tier = tiers.get(c.meta.get("tier"), "?")
            badge = f" · 퀴즈 {len(c.quiz)}" if c.quiz else ""
            ws = widgets_of.get(f"{p.slug}/{c.slug}", [])
            badge += f" · 위젯: {', '.join(ws)}" if ws else ""
            out.append(f"- **{c.label}** [{c.meta.get('title')}]({rel}) · `{tier}`{badge}")
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
    chapters = [c for p in parts for c in p.chapters]
    lines = [f"부 {len(parts)} · 장 {len(chapters)}"]
    for key, names in (("status", config["statuses"]), ("tier", config["tiers"])):
        cnt = Counter(c.meta.get(key) for c in chapters)
        lines.append(f"{key}: " + ", ".join(f"{names.get(k, k)}({k}) {v}" for k, v in cnt.most_common()))
    codes = {t["key"]: t["code"] for t in config["tracks"]}
    track_cnt: Counter = Counter()
    for p in parts:
        track_cnt[p.meta.get("track")] += len(p.chapters)
    lines.append("track: " + ", ".join(f"{codes.get(k, k)} {v}" for k, v in track_cnt.items()))
    quizzed = [c for c in chapters if c.quiz]
    lines.append(f"quiz: {len(quizzed)}개 장 · {sum(len(c.quiz) for c in quizzed)}문항")
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
        if not errors.items and (
            not OUTLINE.exists() or OUTLINE.read_text(encoding="utf-8") != render_outline(parts, config)
        ):
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
