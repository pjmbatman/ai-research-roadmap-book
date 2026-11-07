// 실습실(ADR 0012)이 쓰는 원고: labs/<부>/<장>/ 폴더의 글 파일과 본문 '실습' 절.
import fs from "node:fs";
import path from "node:path";
import { ROOT, type Chapter } from "./book";
import { h2Sections } from "./quest";
import { renderMarkdown } from "./markdown";

const TEXT = new Set([".py", ".md", ".toml", ".txt", ".json", ".csv", ".yaml", ".yml", ".cfg", ".ini"]);
const SKIP = new Set(["__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv"]);

/** 실습 폴더의 파일: 글 파일은 내용, 그 밖(그림 등)은 이름만 */
export function labFiles(ch: Chapter): { text: Record<string, string>; other: string[] } {
  const dir = path.join(ROOT, "labs", ch.partSlug, ch.slug);
  const text: Record<string, string> = {};
  const other: string[] = [];
  const walk = (d: string, rel: string) => {
    if (!fs.existsSync(d)) return;
    for (const name of fs.readdirSync(d).sort()) {
      if (SKIP.has(name) || name.startsWith(".")) continue;
      const full = path.join(d, name), r = rel ? `${rel}/${name}` : name;
      if (fs.statSync(full).isDirectory()) walk(full, r);
      else if (TEXT.has(path.extname(name))) text[r] = fs.readFileSync(full, "utf8");
      else other.push(r);
    }
  };
  walk(dir, "");
  return { text, other };
}

/** 본문 '실습' 절을 안내 문서로(연습문제 자리 표시는 뺀다) */
export function labGuide(ch: Chapter, assetBase: string): string {
  const s = h2Sections(ch.body).find((x) => x.title === "실습");
  if (!s) return "";
  return renderMarkdown(s.text.replace(/^::exercise\{[^}]*\}\s*$/gm, "").trim(), { assetBase, fromDir: ch.dir });
}
