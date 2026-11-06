// 원고의 장별 그림(content/<부>/<장>/figures/*)을 public/content-assets/<부 slug>/<장 slug>/figures/로 복사한다.
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(process.cwd(), "../..");
const CONTENT = path.join(ROOT, "content");
const OUT = path.join(process.cwd(), "public", "content-assets");

fs.rmSync(OUT, { recursive: true, force: true });
let n = 0;
for (const p of fs.readdirSync(CONTENT)) {
  const pdir = path.join(CONTENT, p);
  if (!fs.statSync(pdir).isDirectory()) continue;
  const pslug = p === "appendices" ? p : p.replace(/^\d{2}-/, "");
  for (const c of fs.readdirSync(pdir)) {
    const fdir = path.join(pdir, c, "figures");
    if (!fs.existsSync(fdir)) continue;
    const cslug = c.replace(/^([0-9]{2}|[a-z])-/, "");
    const dest = path.join(OUT, pslug, cslug, "figures");
    fs.mkdirSync(dest, { recursive: true });
    for (const f of fs.readdirSync(fdir)) { fs.copyFileSync(path.join(fdir, f), path.join(dest, f)); n++; }
  }
}
console.log(`sync-assets: ${n}개 그림 복사`);
