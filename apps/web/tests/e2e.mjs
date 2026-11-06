// 브라우저 종단 테스트(Playwright). 빌드된 dist/를 작은 정적 서버로 띄우고 실제 Chromium에서 확인한다.
// 사용: PYODIDE_DIR=<pyodide 배포본 폴더> node tests/e2e.mjs
//   - 빌드는 PUBLIC_PYODIDE_URL=/pyodide/ 로 해 두어야 한다. 이 스크립트가 PYODIDE_DIR을 /pyodide/ 경로로 서빙한다.
//   - PYODIDE_DIR이 없으면 코드 실행·채점 테스트는 건너뛴다(퀴즈·위젯·검색·진도만 확인).
//   - SHOTS=<폴더>를 주면 주요 화면의 스크린샷을 저장한다.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// 프로젝트에 설치된 playwright가 없으면 전역(NODE_PATH) 설치를 쓴다.
const { chromium } = await import("playwright").catch(() => createRequire(import.meta.url)("playwright"));

const DIST = path.resolve(process.env.DIST || "dist");
const PYO = process.env.PYODIDE_DIR ? path.resolve(process.env.PYODIDE_DIR) : null;
const SHOTS = process.env.SHOTS ? path.resolve(process.env.SHOTS) : null;
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".wasm": "application/wasm", ".zip": "application/zip", ".whl": "application/zip", ".woff2": "font/woff2", ".woff": "font/woff", ".tar": "application/x-tar" };

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = u.startsWith("/pyodide/") && PYO ? path.join(PYO, u.slice("/pyodide/".length)) : path.join(DIST, u);
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!fs.existsSync(file)) { res.writeHead(404); res.end("not found"); return; }
  res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}`;

const results = [];
const errors = [];
async function test(name, fn) {
  try { await fn(); results.push([true, name]); console.log(`  ✓ ${name}`); }
  catch (e) { results.push([false, name]); console.log(`  ✗ ${name}\n    ${String(e.message || e).split("\n").slice(0, 4).join("\n    ")}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m); };

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
page.on("console", (m) => { if (m.type() === "error") errors.push(`${page.url()} :: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`${page.url()} :: ${e.message}`));
const go = (p) => page.goto(BASE + p, { waitUntil: "networkidle" });
const shot = async (name, fullPage = false) => { if (SHOTS) { fs.mkdirSync(SHOTS, { recursive: true }); await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage }); } };

const CH72 = "/linear-algebra/affine-and-nonlinear/";
const CH123 = "/statistical-inference/distributional-formulation/";

console.log("홈·검색·테마");
await test("홈: 레이어 스택이 레이어를 고르면 카드가 바뀐다", async () => {
  await go("/");
  await page.click('.stack-legend button[data-level="3"]');
  const t = await page.textContent("[data-stack-title]");
  assert(t.includes("L3"), `카드 제목: ${t}`);
  assert((await page.locator("[data-stack-tracks] a").count()) >= 1, "L3 트랙 링크 없음");
  await shot("home");
});
await test("⌘K 검색: 장 번호로 찾아 이동한다", async () => {
  await page.keyboard.press("Control+K");
  await page.fill("#palette-input", "12.3");
  await page.waitForSelector('.palette li[aria-selected="true"] a');
  await Promise.all([page.waitForNavigation(), page.keyboard.press("Enter")]);
  assert(page.url().endsWith(CH123), `이동한 주소: ${page.url()}`);
});
await test("테마 토글이 data-theme을 바꾸고 저장한다", async () => {
  const before = await page.evaluate(() => document.documentElement.dataset.theme || "");
  await page.click("[data-theme-toggle]");
  const after = await page.evaluate(() => document.documentElement.dataset.theme);
  assert(after && after !== before, `${before} → ${after}`);
  assert((await page.evaluate(() => localStorage.getItem("roadmap:theme"))) === after, "저장 안 됨");
  await page.evaluate(() => { localStorage.removeItem("roadmap:theme"); delete document.documentElement.dataset.theme; });
});

console.log("퀴즈 (7.2)");
await go(CH72);
await test("단일 선택: 오답은 피드백, 정답은 해설과 진도 기록", async () => {
  const q = page.locator("#quiz-affine-is-not-linear");
  await q.locator(".quiz-choice").nth(0).click();
  await q.locator(".quiz-choice.is-wrong").waitFor();
  assert((await q.locator(".quiz-fb").textContent()).includes("원점"), "오답 피드백이 선택지 feedback이 아님");
  await q.locator(".quiz-choice").nth(1).click();
  await q.locator(".quiz-fb.good").waitFor();
  const p = await page.evaluate(() => JSON.parse(localStorage.getItem("roadmap:progress:v1")).chapters["linear-algebra/affine-and-nonlinear"].quiz["affine-is-not-linear"]);
  assert(p.correct && p.tries === 2, JSON.stringify(p));
});
await test("복수 선택: 일부만 고르면 부족하다고 알려 주고, 모두 고르면 정답", async () => {
  const q = page.locator("#quiz-affine-combinations");
  await q.locator(".quiz-choice").nth(0).click();
  await q.locator("button:has-text('확인')").click();
  assert((await q.locator(".quiz-fb").textContent()).includes("더 골라야"), "누락 안내 없음");
  await q.locator(".quiz-choice").nth(1).click();
  await q.locator(".quiz-choice").nth(2).click();
  await q.locator("button:has-text('확인')").click();
  await q.locator(".quiz-fb.good").waitFor();
});
await test("수치 문항: 슬라이더로 답을 고른다", async () => {
  const q = page.locator("#quiz-relu-linear-regions");
  const r = q.locator('input[type="range"]');
  await r.fill("3");
  await q.locator("button:has-text('확인')").click();
  assert(!(await q.locator(".quiz-fb").getAttribute("class")).includes("good"), "3은 오답이어야 함");
  await r.fill("4");
  assert((await q.locator("output").textContent()).trim() === "4", "슬라이더 출력 갱신 안 됨");
  await q.locator("button:has-text('확인')").click();
  await q.locator(".quiz-fb.good").waitFor();
});
await test("개념 퀴즈 요약이 학습 목표별 진행을 보여 준다", async () => {
  const s = await page.textContent(".quiz-summary .sum-head");
  assert(/3 \/ 6/.test(s), s);
});
await test("통과 기준 체크가 진도에 저장되고 진도 막대가 움직인다", async () => {
  await page.locator(".box-pass li .check").first().check();
  const w = await page.evaluate(() => document.querySelector("[data-chapter-meter]").style.width);
  assert(w && w !== "0%", `막대 ${w}`);
});

console.log("위젯");
await test("linear-map-2d: bias를 끄면 선형사상이 된다", async () => {
  const w = page.locator('.widget[data-widget="linear-map-2d"]');
  await w.scrollIntoViewIfNeeded();
  await page.waitForSelector('.widget[data-widget="linear-map-2d"][data-ready="true"]');
  assert((await w.locator(".readouts").textContent()).includes("아핀 사상"), "처음에는 아핀");
  await w.locator(".toggle input").uncheck();
  assert((await w.locator(".readouts").textContent()).includes("선형사상"), "bias를 끄면 선형");
  await w.locator(".preset", { hasText: "특이" }).click();
  assert((await w.locator(".readouts").textContent()).includes("det A = 0.00"), "특이 행렬 det");
  await shot("widget-linear-map");
});
await test("relu-regions: 뉴런을 늘리면 영역 수가 는다", async () => {
  const w = page.locator('.widget[data-widget="relu-regions"]');
  await w.scrollIntoViewIfNeeded();
  await page.waitForSelector('.widget[data-widget="relu-regions"][data-ready="true"]');
  const count = async () => Number((await w.locator(".readout.hot").textContent()).match(/(\d+)/)[1]);
  await w.locator('input[type="range"]').fill("1");
  const one = await count();
  await w.locator('input[type="range"]').fill("12");
  const many = await count();
  assert(one === 2 && many > 20, `1개: ${one}, 12개: ${many}`);
});
await go(CH123);
await test("loss-to-estimator: 지수분포에서 손실마다 다른 대표값", async () => {
  const w = page.locator('.widget[data-widget="loss-to-estimator"]');
  await w.scrollIntoViewIfNeeded();
  await page.waitForSelector('.widget[data-widget="loss-to-estimator"][data-ready="true"]');
  const t = await w.locator(".readouts").textContent();
  const num = (re) => Number(t.match(re)[1]);
  const mean = num(/평균 ([\d.]+)/), med = num(/중앙값 ([\d.]+)/), q90 = num(/90% 분위수 ([\d.]+)/);
  assert(Math.abs(mean - 1) < 0.03 && Math.abs(med - Math.LN2) < 0.03 && Math.abs(q90 - Math.log(10)) < 0.03, t);
  await w.locator(".preset", { hasText: "쌍봉" }).click();
  assert((await w.locator(".readouts").textContent()).includes("평균 0.00") || (await w.locator(".readouts").textContent()).includes("평균 -0.00"), "쌍봉 평균 0");
  await shot("widget-loss");
});

if (PYO) {
  console.log("브라우저 Python (Pyodide)");
  await go(CH72);
  await test("코드 셀 실행: 기록된 출력과 같은 결과", async () => {
    const cell = page.locator(".cell").first();
    await cell.scrollIntoViewIfNeeded();
    await cell.locator("button.run").click();
    await cell.locator('.cell-out[data-live="true"]').waitFor({ timeout: 120000 });
    await page.waitForFunction(() => document.querySelector(".cell .cell-status")?.textContent?.startsWith("완료") || document.querySelector(".cell .cell-status")?.textContent === "오류", null, { timeout: 120000 });
    const out = await cell.locator(".cell-out").textContent();
    assert(out.includes("f(0)             = [0.5  0.25]"), out);
  });
  await test("같은 session의 뒤 셀만 실행해도 앞 셀을 먼저 실행한다", async () => {
    await page.reload({ waitUntil: "networkidle" });
    const third = page.locator(".cell").nth(2);
    await third.scrollIntoViewIfNeeded();
    await third.locator("button.run").click();
    await page.waitForFunction(() => /완료|오류/.test(document.querySelectorAll(".cell .cell-status")[2]?.textContent || ""), null, { timeout: 120000 });
    const out = await third.locator(".cell-out").textContent();
    assert(out.includes("True") && out.includes("rank(W) = 2"), out);
  });
  await test("연습문제: 시작 코드는 테스트에 실패한다", async () => {
    const ex = page.locator('.exercise[data-exercise="fold-affine-layers"]');
    await ex.scrollIntoViewIfNeeded();
    await ex.locator("[data-run]").click();
    await page.waitForFunction(() => /통과|오류/.test(document.querySelector('.exercise[data-exercise="fold-affine-layers"] .cell-status')?.textContent || ""), null, { timeout: 120000 });
    assert((await ex.locator(".ex-results .fail").count()) > 0, "실패가 있어야 함");
  });
  await test("연습문제: 해답 코드는 모든 테스트를 통과하고 진도에 기록된다", async () => {
    const sol = await page.evaluate(() => JSON.parse(document.getElementById("page-data").textContent).exercises["relu-breakpoints"].solution);
    await page.evaluate((code) => {
      const k = "roadmap:progress:v1"; const s = JSON.parse(localStorage.getItem(k));
      (s.chapters["linear-algebra/affine-and-nonlinear"].exercises ??= {})["relu-breakpoints"] = { passed: false, code };
      localStorage.setItem(k, JSON.stringify(s));
    }, sol);
    await page.reload({ waitUntil: "networkidle" });
    const ex = page.locator('.exercise[data-exercise="relu-breakpoints"]');
    await ex.scrollIntoViewIfNeeded();
    await ex.locator("[data-run]").click();
    await page.waitForFunction(() => /통과|오류/.test(document.querySelector('.exercise[data-exercise="relu-breakpoints"] .cell-status')?.textContent || ""), null, { timeout: 120000 });
    const status = await ex.locator(".cell-status").textContent();
    const fails = await ex.locator(".ex-results .fail").count();
    assert(fails === 0 && /^(\d+) \/ \1 통과$/.test(status), `${status}, 실패 ${fails}: ${await ex.locator(".ex-results").textContent()}`);
    assert((await ex.locator(".ex-state").textContent()).includes("통과"), "상태 배지");
    await shot("exercise");
  });
}

console.log("진도·지도·트랙");
await test("진도 페이지가 기록을 요약한다", async () => {
  await go("/progress/");
  const v = Number(await page.textContent('[data-k="visited"]'));
  const q = Number(await page.textContent('[data-k="quiz"]'));
  assert(v >= 2 && q >= 3, `읽은 장 ${v}, 맞힌 퀴즈 ${q}`);
  assert((await page.locator('[data-k="recent"] a').count()) >= 2, "최근 장");
  await shot("progress");
});
await test("지도: 장을 누르면 정보 카드에 선수 지식 수가 나온다", async () => {
  await go("/map/");
  const box = await page.locator("[data-map] canvas").boundingBox();
  // 화면 좌표를 훑어 첫 노드를 찾는다
  const hit = await page.evaluate(() => {
    const c = document.querySelector("[data-map] canvas");
    return { w: c.clientWidth, h: c.clientHeight };
  });
  let found = false;
  for (let y = 95; y < hit.h && !found; y += 6) {
    for (let x = 20; x < hit.w && !found; x += hit.w / 12 / 2) {
      await page.mouse.move(box.x + x, box.y + y);
      if ((await page.textContent("[data-map-info]")).includes("선수 지식 전체")) found = true;
    }
  }
  assert(found, "노드를 찾지 못함");
  await shot("map");
});
await test("트랙 페이지가 읽은 장 수를 보여 준다", async () => {
  await go("/tracks/mathematical-foundations/");
  assert(Number(await page.textContent("[data-track-done]")) >= 1, "읽은 장 0");
});
await test("모바일 폭에서 장 페이지가 가로로 넘치지 않는다", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  await go(CH123);
  const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert(over <= 0, `가로 넘침 ${over}px`);
  await shot("mobile-chapter");
});

await browser.close();
server.close();
const real = errors.filter((e) => !/favicon/.test(e));
if (real.length) { console.log(`\n콘솔 오류 ${real.length}개:\n  ${real.slice(0, 15).join("\n  ")}`); }
const failed = results.filter(([ok]) => !ok).length;
console.log(`\n${results.length - failed}/${results.length} 통과, 콘솔 오류 ${real.length}개`);
process.exit(failed || real.length ? 1 : 0);
