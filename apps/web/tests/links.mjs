// 빌드 결과(dist/)의 모든 HTML에서 내부 링크·자원 경로가 실제 파일을 가리키는지 확인한다.
// 사용: node tests/links.mjs [dist 경로] [base 경로]   예) node tests/links.mjs dist /ai-research-roadmap-book/
import fs from "node:fs";
import path from "node:path";

const dist = path.resolve(process.argv[2] || "dist");
const base = process.argv[3] || "/";
const files = [];
(function walk(d) { for (const f of fs.readdirSync(d)) { const p = path.join(d, f); fs.statSync(p).isDirectory() ? walk(p) : files.push(p); } })(dist);
const html = files.filter((f) => f.endsWith(".html"));
const ids = new Map();
const bad = [];
let checked = 0;
for (const f of html) {
  const text = fs.readFileSync(f, "utf8");
  ids.set(f, new Set([...text.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
}
for (const f of html) {
  const text = fs.readFileSync(f, "utf8");
  for (const m of text.matchAll(/\s(?:href|src)="([^"]+)"/g)) {
    let url = m[1].replace(/&amp;/g, "&");
    if (/^(https?:|mailto:|data:|javascript:)/.test(url)) continue;
    checked++;
    const [p, hash] = url.split("#");
    let target;
    if (p === "") target = f;
    else if (p.startsWith("/")) {
      if (!p.startsWith(base)) { bad.push(`${path.relative(dist, f)} → ${url} (base 밖)`); continue; }
      target = path.join(dist, decodeURIComponent(p.slice(base.length)));
    } else target = path.join(path.dirname(f), decodeURIComponent(p));
    if (fs.existsSync(target) && fs.statSync(target).isDirectory()) target = path.join(target, "index.html");
    if (!fs.existsSync(target)) { bad.push(`${path.relative(dist, f)} → ${url}`); continue; }
    if (hash && target.endsWith(".html") && !ids.get(target)?.has(decodeURIComponent(hash))) bad.push(`${path.relative(dist, f)} → ${url} (앵커 없음)`);
  }
}
console.log(`HTML ${html.length}개, 내부 링크 ${checked}개 확인`);
if (bad.length) { console.log(`깨진 링크 ${bad.length}개`); console.log([...new Set(bad)].slice(0, 40).join("\n")); process.exit(1); }
console.log("깨진 링크 0개");
