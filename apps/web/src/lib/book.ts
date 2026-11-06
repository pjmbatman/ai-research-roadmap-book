// 빌드 시점에 저장소의 원고(content/, book.toml, data/, labs/)를 읽는다. 원고가 유일한 원천이다(ADR 0001).
import fs from "node:fs";
import path from "node:path";
import { parse as parseToml } from "smol-toml";

export const ROOT = path.resolve(process.cwd(), "../..");
const CONTENT = path.join(ROOT, "content");
const APPENDIX = "appendices";

export type Meta = Record<string, string | string[]>;

export interface Track {
  key: string; code: string; name: string; name_en: string; layer: number;
  question: string; outcome: string; assessment: string[]; capstone: string;
}
export interface Layer { level: number; name: string; name_en: string; description: string }

export interface QuizChoice { text: string; correct?: boolean; feedback?: string }
export interface QuizQuestion {
  id: string; type: "single" | "multiple" | "numeric"; objective: number; prompt: string; explanation: string;
  choices?: QuizChoice[]; answer?: number; tolerance?: number;
  slider?: { min: number; max: number; step: number; unit?: string };
}
export interface Exercise {
  id: string; title: string; exec: string; entry: string; objective: number;
  problem: string; starter: string; tests: string; solution: string;
}
export interface Chapter {
  id: string; slug: string; partSlug: string; order: string; label: string; meta: Meta; body: string;
  quiz: QuizQuestion[]; exercises: Record<string, Exercise>; dir: string; url: string;
}
export interface Part {
  slug: string; num: string; label: string; meta: Meta; body: string; chapters: Chapter[]; url: string; dir: string;
}
export interface Widget { name: string; title: string; kind: string; chapters: string[]; status: string; interaction: string; binds?: string[]; motion?: string }

function parseValue(raw: string): string | string[] {
  raw = raw.trim();
  if (raw.startsWith("[") && raw.endsWith("]")) return raw.slice(1, -1).split(",").map((s) => s.trim()).filter(Boolean);
  if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) return raw.slice(1, -1).replace(/\\"/g, '"');
  return raw;
}

export function readDoc(file: string): { meta: Meta; body: string } {
  const text = fs.readFileSync(file, "utf8");
  if (!text.startsWith("---\n")) return { meta: {}, body: text };
  const end = text.indexOf("\n---\n", 4);
  const meta: Meta = {};
  for (const line of text.slice(4, end).split("\n")) {
    const i = line.indexOf(":");
    if (i < 0 || line.trim().startsWith("#")) continue;
    meta[line.slice(0, i).trim()] = parseValue(line.slice(i + 1));
  }
  return { meta, body: text.slice(end + 5) };
}

const s = (v: string | string[] | undefined) => (Array.isArray(v) ? v.join(", ") : v ?? "");
export const str = s;

function readExercises(partSlug: string, chSlug: string): Record<string, Exercise> {
  const dir = path.join(ROOT, "labs", partSlug, chSlug, "exercises");
  const out: Record<string, Exercise> = {};
  if (!fs.existsSync(dir)) return out;
  for (const id of fs.readdirSync(dir).sort()) {
    const d = path.join(dir, id);
    if (!fs.statSync(d).isDirectory()) continue;
    const read = (f: string) => (fs.existsSync(path.join(d, f)) ? fs.readFileSync(path.join(d, f), "utf8") : "");
    const cfg = parseToml(read("exercise.toml") || "") as Record<string, unknown>;
    out[id] = {
      id, title: String(cfg.title ?? id), exec: String(cfg.exec ?? "browser"), entry: String(cfg.entry ?? ""),
      objective: Number(cfg.objective ?? 1), problem: read("problem.md"), starter: read("starter.py"),
      tests: read("test_exercise.py"), solution: read("solution.py"),
    };
  }
  return out;
}

let cache: { parts: Part[]; config: any; widgets: Widget[] } | null = null;

export function loadBook() {
  if (cache) return cache;
  const config = parseToml(fs.readFileSync(path.join(ROOT, "book.toml"), "utf8")) as any;
  const wfile = path.join(ROOT, "data", "widgets.toml");
  const widgets = (fs.existsSync(wfile) ? (parseToml(fs.readFileSync(wfile, "utf8")) as any).widgets : []) as Widget[];
  const parts: Part[] = [];
  for (const pdirName of fs.readdirSync(CONTENT).sort()) {
    const pdir = path.join(CONTENT, pdirName);
    if (!fs.statSync(pdir).isDirectory()) continue;
    const isApp = pdirName === APPENDIX;
    const m = /^(\d{2})-(.+)$/.exec(pdirName);
    if (!isApp && !m) continue;
    const slug = isApp ? APPENDIX : m![2];
    const label = isApp ? "부록" : String(Number(m![1]));
    const { meta, body } = readDoc(path.join(pdir, "index.md"));
    const part: Part = { slug, num: isApp ? "ap" : m![1], label, meta, body, chapters: [], url: `${slug}/`, dir: pdir };
    for (const cdirName of fs.readdirSync(pdir).sort()) {
      const cdir = path.join(pdir, cdirName);
      if (!fs.statSync(cdir).isDirectory()) continue;
      const cm = /^([0-9]{2}|[a-z])-(.+)$/.exec(cdirName);
      if (!cm) continue;
      const doc = readDoc(path.join(cdir, "index.md"));
      const qfile = path.join(cdir, "quiz.toml");
      const quiz = fs.existsSync(qfile) ? ((parseToml(fs.readFileSync(qfile, "utf8")) as any).questions ?? []) : [];
      part.chapters.push({
        id: `${slug}/${cm[2]}`, slug: cm[2], partSlug: slug, order: cm[1],
        label: isApp ? cm[1].toUpperCase() : `${label}.${Number(cm[1])}`,
        meta: doc.meta, body: doc.body, quiz, exercises: readExercises(slug, cm[2]), dir: cdir, url: `${slug}/${cm[2]}/`,
      });
    }
    parts.push(part);
  }
  cache = { parts, config, widgets };
  return cache;
}

export function allChapters(): Chapter[] {
  return loadBook().parts.flatMap((p) => p.chapters);
}
export function chapterById(id: string) {
  return allChapters().find((c) => c.id === id);
}
export function trackByKey(key: string): Track | undefined {
  return (loadBook().config.tracks as Track[]).find((t) => t.key === key);
}
export function partOf(ch: Chapter): Part {
  return loadBook().parts.find((p) => p.slug === ch.partSlug)!;
}
export function trackOfChapter(ch: Chapter): Track | undefined {
  return trackByKey(s(partOf(ch).meta.track));
}
export function prerequisites(ch: Chapter): Chapter[] {
  const ids = (ch.meta.prerequisites as string[]) || [];
  return ids.map(chapterById).filter(Boolean) as Chapter[];
}
export function dependents(ch: Chapter): Chapter[] {
  return allChapters().filter((c) => ((c.meta.prerequisites as string[]) || []).includes(ch.id));
}
export function neighbors(ch: Chapter): { prev?: Chapter; next?: Chapter } {
  const all = allChapters();
  const i = all.findIndex((c) => c.id === ch.id);
  return { prev: all[i - 1], next: all[i + 1] };
}
export function widgetsFor(ch: Chapter): Widget[] {
  return loadBook().widgets.filter((w) => w.chapters.includes(ch.id));
}
export function headings(body: string): string[] {
  return [...body.matchAll(/^## (.+?)\s*$/gm)].map((m) => m[1]);
}
export const FIXED = ["학습 목표", "실습", "실패 사례", "통과 기준"];
export const STATUS_LABEL: Record<string, string> = { outline: "목차", spec: "명세", draft: "초고", review: "검토", done: "완료" };
export const TIER_LABEL: Record<string, string> = { core: "필수", advanced: "심화", reference: "참조" };
