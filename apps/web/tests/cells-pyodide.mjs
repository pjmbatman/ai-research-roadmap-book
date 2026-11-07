// 모든 장의 exec="browser" 셀을 Pyodide(Node)에서 장마다 순서대로 실행해 본문 output 블록과 비교한다.
// 브라우저에서 독자가 보는 출력이 본문과 같은지 확인한다(로컬 CPython 출력은 book.py check와 cells.py가 확인한다).
// 사용: node tests/cells-pyodide.mjs [저장소 루트] [장 경로 필터...]
// PYODIDE_DIR을 주면 그 폴더의 Pyodide를 쓰고, 없으면 npm 패키지를 쓴다(빠진 패키지는 CDN에서 받아 캐시한다).
import fs from "node:fs";
import path from "node:path";
const repo = path.resolve(process.argv[2] || "../..");
const filters = process.argv.slice(3);
const PYO = process.env.PYODIDE_DIR ? path.resolve(process.env.PYODIDE_DIR) + "/" : null;
const { loadPyodide } = PYO ? await import(path.join(PYO, "pyodide.mjs")) : await import("pyodide");
const py = await loadPyodide(PYO ? { indexURL: PYO } : {});
let buf = [];
py.setStdout({ batched: (s) => buf.push(s + "\n") });
py.setStderr({ batched: (s) => buf.push(s + "\n") });

const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : e.name === "index.md" ? [path.join(d, e.name)] : []);
const files = walk(path.join(repo, "content")).filter((f) => !filters.length || filters.some((x) => f.includes(x))).sort();
const RE = /```python \{([^}]*exec="browser"[^}]*)\}\n([\s\S]*?)\n```\s*\n(?:```output\n([\s\S]*?)\n```)?/g;
let total = 0, bad = 0;
const norm = (s) => s.replace(/[ \t]+$/gm, "").replace(/\n+$/, "");
for (const f of files) {
  const md = fs.readFileSync(f, "utf8");
  const spaces = new Map();
  let m, i = 0;
  RE.lastIndex = 0;
  while ((m = RE.exec(md))) {
    i++;
    const sess = (/session="([^"]+)"/.exec(m[1]) || [, "default"])[1];
    if (!spaces.has(sess)) { const d = py.globals.get("dict")(); d.set("__name__", "__main__"); spaces.set(sess, d); }
    buf = [];
    let err = "", t0 = Date.now();
    try {
      await py.loadPackagesFromImports(m[2], { messageCallback: () => {}, errorCallback: (e) => { err ||= "패키지 불러오기 실패: " + e; } });
      t0 = Date.now(); // 패키지 내려받기·불러오기 시간은 빼고 실행 시간만 잰다
      await py.runPythonAsync(m[2], { globals: spaces.get(sess) });
    } catch (e) { err = String(e.message || e).split("\n").slice(-3).join(" | "); }
    const dt = Date.now() - t0;
    total++;
    const got = norm(buf.join("")), want = m[3] === undefined ? null : norm(m[3]);
    const rel = path.relative(repo, f).replace("/index.md", "");
    if (err || (want !== null && got !== want) || dt > 8000) {
      bad++;
      console.log(`✗ ${rel} cell ${i} (${dt}ms)${err ? " ERROR " + err : ""}`);
      if (!err && want !== null && got !== want) {
        const gl = got.split("\n"), wl = want.split("\n");
        for (let k = 0; k < Math.max(gl.length, wl.length); k++) if (gl[k] !== wl[k]) { console.log(`   want: ${wl[k]}\n   got:  ${gl[k]}`); break; }
      }
    }
  }
}
console.log(`\n${total - bad}/${total} 셀 일치`);
process.exit(bad ? 1 : 0);
