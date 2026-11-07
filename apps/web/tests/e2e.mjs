// 브라우저 종단 테스트(Playwright). 빌드된 dist/를 작은 정적 서버로 띄우고 실제 Chromium에서 확인한다.
// 사용: PYODIDE_DIR=<pyodide 배포본 폴더> node tests/e2e.mjs
//   - 빌드는 PUBLIC_PYODIDE_URL=/pyodide/ 로 해 두어야 한다. 이 스크립트가 PYODIDE_DIR을 /pyodide/ 경로로 서빙한다.
//   - PYODIDE_DIR이 없으면 코드 실행·채점 테스트는 건너뛴다(퀴즈·위젯·검색·진도만 확인).
//     PYODIDE_CDN=1이면 기본 빌드(CDN의 Pyodide)로 코드 실행·채점까지 확인한다(CI용, 네트워크 필요).
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
const over = () => page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

console.log("메인·검색·테마");
await test("메인(Bento): 타일 일곱 개, 이어서 학습 타일이 본문 있는 장으로 간다", async () => {
  await go("/");
  assert((await page.locator(".bento > .tile").count()) === 7, `타일 ${await page.locator(".bento > .tile").count()}`);
  const href = await page.getAttribute("[data-hero]", "href");
  assert(/^\/[a-z0-9-]+\/[a-z0-9-]+\/$/.test(href), `이어서 학습이 장을 가리켜야 함: ${href}`);
  assert((await page.locator("[data-hero] .mini-path i").count()) >= 10, "진행 점");
  assert((await over()) <= 0, "가로 넘침");
  await shot("home");
});
await test("⌘K 검색: 장 번호로 찾아 이동한다", async () => {
  await page.keyboard.press("Control+K");
  await page.fill("#palette-input", "12.3");
  await page.waitForSelector('.palette li[aria-selected="true"] a');
  await Promise.all([page.waitForNavigation(), page.keyboard.press("Enter")]);
  assert(new URL(page.url()).pathname === CH123, `이동한 주소: ${page.url()}`);
});
await test("테마 토글이 data-theme을 바꾸고 저장한다", async () => {
  const before = await page.evaluate(() => document.documentElement.dataset.theme || "");
  await page.click("[data-theme-toggle]");
  const after = await page.evaluate(() => document.documentElement.dataset.theme);
  assert(after && after !== before, `${before} → ${after}`);
  assert((await page.evaluate(() => localStorage.getItem("roadmap:theme"))) === after, "저장 안 됨");
  await page.evaluate(() => { localStorage.removeItem("roadmap:theme"); delete document.documentElement.dataset.theme; });
});

// 퀘스트 도우미: 노드가 있는 구간 탭을 고르고 노드를 눌러 시트를 연다
const openNode = async (id) => {
  const gi = await page.evaluate((id) => JSON.parse(document.getElementById("quest-data").textContent).groups.findIndex((g) => g.nodes.includes(id)), id);
  assert(gi >= 0, `노드 없음: ${id}`);
  if (await page.locator(".sec-tabs button").count()) await page.locator(".sec-tabs button").nth(gi).click();
  await page.locator(`.node[data-node="${id}"]`).click({ force: true });
  await page.locator(".ls-bg.on").waitFor();
};
const closeSheet = async () => { await page.keyboard.press("Escape"); await page.waitForFunction(() => !document.querySelector(".ls-bg")); };
// 시트에서 그 요소가 있는 쪽까지 넘긴다(위젯은 화면에 들어온 뒤에 그려지므로 쪽 표시로 판단한다)
const onPage = (sel) => page.evaluate((sel) => [...document.querySelectorAll(`.ls ${sel}`)].some((el) => !el.closest(".pg-off")), sel);
const pageTo = async (sel) => {
  for (let i = 0; i < 12; i++) {
    if (await onPage(sel)) return;
    if ((await page.textContent(".ls [data-next]")).trim() === "완료") break;
    await page.locator(".ls [data-next]").click();
  }
  if (!(await onPage(sel))) throw new Error(`쪽을 찾지 못함: ${sel}`);
};

console.log("퀘스트 (7.2)");
await go(CH72);
await test("퀘스트: 구간 탭과 노드, 지금 할 노드에 말풍선", async () => {
  const n = await page.evaluate(() => JSON.parse(document.getElementById("quest-data").textContent).nodes.length);
  assert((await page.locator(".node").count()) === n && n >= 12, `노드 ${n}`);
  assert((await page.locator(".sec-tabs button").count()) >= 3, "구간 탭");
  assert((await page.locator(".node.cur .tag").textContent()).includes("시작"), "시작 말풍선");
  await page.locator(".sec-tabs button").nth(1).click();
  const hidden = (g) => page.locator(`.path-group[data-group="${g}"]`).evaluate((e) => e.hidden);
  assert(!(await hidden(1)) && (await hidden(0)), "두 번째 구간만 보여야 함");
  assert(await page.locator('.path-group[data-group="1"] .node').first().isVisible(), "두 번째 구간의 노드");
  await page.locator(".sec-tabs button").nth(0).click();
  await shot("quest");
});
await test("읽기 노드: 시트에서 완료하면 노드가 끝남으로 바뀌고 다음 노드로 넘어간다", async () => {
  await openNode("s:학습-목표");
  await page.locator(".ls [data-next]").click();
  await page.waitForFunction(() => !document.querySelector(".ls-bg"));
  assert(await page.locator('.node[data-node="s:학습-목표"]').evaluate((b) => b.classList.contains("done")), "끝남 표시");
  assert(await page.locator('.node[data-node="s:선형-변환과-아핀-변환"]').evaluate((b) => b.classList.contains("cur")), "다음 노드");
  assert((await page.textContent("[data-q-count]")).trim().startsWith("1 /"), await page.textContent("[data-q-count]"));
});
await test("시트는 한 쪽씩 넘기고, 주소의 #이 절을 가리킨다", async () => {
  await openNode("s:선형-변환과-아핀-변환");
  const k = async () => (await page.textContent(".ls-k")).trim();
  assert(/^1 \/ [3-9]$/.test(await k()), await k());
  assert(decodeURIComponent(page.url()).endsWith("#선형-변환과-아핀-변환"), page.url());
  await page.locator(".ls [data-next]").click();
  assert((await k()).startsWith("2 /"), await k());
  await page.keyboard.press("ArrowRight");
  assert((await k()).startsWith("3 /"), "→ 키");
  await page.keyboard.press("ArrowLeft");
  assert((await k()).startsWith("2 /"), "← 키");
  await shot("sheet");
  await closeSheet();
  assert(!page.url().includes("#"), "닫으면 # 제거");
});
await test("주소의 #으로 그 노드의 시트를 바로 연다", async () => {
  await go(CH72 + "#" + encodeURIComponent("통과-기준"));
  await page.locator(".ls-bg.on .boss-list").waitFor();
  assert((await page.locator(".boss-row").count()) >= 2, "통과 기준 줄");
  await closeSheet();
});

console.log("퀴즈 (7.2)");
await test("퀴즈 노드: 오답 피드백, 다시, 정답 기록", async () => {
  await openNode("q:affine-is-not-linear");
  const q = page.locator(".ls #quiz-affine-is-not-linear");
  assert(await q.locator(".quiz-primary").isDisabled(), "답을 고르기 전에는 확인 버튼이 비활성");
  await q.locator(".quiz-choice").nth(0).click();
  await q.locator(".quiz-primary").click();
  await q.locator(".quiz-choice.is-wrong").waitFor();
  assert((await q.locator(".quiz-fb").textContent()).includes("원점"), "오답 피드백이 선택지 feedback이 아님");
  assert((await q.locator(".quiz-primary").textContent()).trim() === "다시", "오답 뒤 주 버튼은 '다시'");
  await q.locator(".quiz-choice").nth(1).click();
  await q.locator(".quiz-primary").click();
  await q.locator(".quiz-fb.good").waitFor();
  await shot("quiz");
  await q.locator(".quiz-primary").click();   // 계속 → 시트 닫힘
  await page.waitForFunction(() => !document.querySelector(".ls-bg"));
  assert(await page.locator('.node[data-node="q:affine-is-not-linear"]').evaluate((b) => b.classList.contains("done")), "퀴즈 노드 끝남");
  const p = await page.evaluate(() => JSON.parse(localStorage.getItem("roadmap:progress:v1")).chapters["linear-algebra/affine-and-nonlinear"].quiz["affine-is-not-linear"]);
  assert(p.correct && p.tries === 2, JSON.stringify(p));
});
await test("복수 선택은 누락을 알려 준다", async () => {
  await openNode("q:affine-combinations");
  const q = page.locator(".ls #quiz-affine-combinations");
  await q.locator(".quiz-choice").nth(0).click();
  await q.locator(".quiz-primary").click();
  assert((await q.locator(".quiz-fb").textContent()).includes("더 골라야"), "누락 안내 없음");
  await q.locator(".quiz-primary").click();
  await q.locator(".quiz-choice").nth(1).click();
  await q.locator(".quiz-choice").nth(2).click();
  await q.locator(".quiz-primary").click();
  await q.locator(".quiz-fb.good").waitFor();
  await closeSheet();
});
await test("수치 문항: 슬라이더로 답을 고른다", async () => {
  await openNode("q:relu-linear-regions");
  const q = page.locator(".ls #quiz-relu-linear-regions");
  const r = q.locator('input[type="range"]');
  await r.fill("3");
  await q.locator(".quiz-primary").click();
  assert(!(await q.locator(".quiz-fb").getAttribute("class")).includes("good"), "3은 오답이어야 함");
  await q.locator(".quiz-primary").click();
  await r.fill("4");
  assert((await q.locator("output").textContent()).trim() === "4", "슬라이더 출력 갱신 안 됨");
  await q.locator(".quiz-primary").click();
  await q.locator(".quiz-fb.good").waitFor();
  await closeSheet();
});
await test("통과 기준: 자기 점검 체크가 저장되고 줄이 갱신된다", async () => {
  await openNode("s:통과-기준");
  await page.locator(".ls .box-pass li .check").first().check();
  const row = await page.locator(".boss-row").last().textContent();
  assert(row.includes("1/"), row);
  const c = await page.evaluate(() => JSON.parse(localStorage.getItem("roadmap:progress:v1")).chapters["linear-algebra/affine-and-nonlinear"].checks);
  assert(c && c.c1 === true, JSON.stringify(c));
  assert((await page.textContent(".ls [data-next]")).includes("퀴즈"), "남은 퀴즈로 안내");
  await shot("boss");
  await closeSheet();
});

console.log("위젯 (시트 안)");
await test("linear-map-2d: bias를 끄면 선형사상이 된다", async () => {
  await openNode("s:선형-변환과-아핀-변환");
  await pageTo('.widget[data-widget="linear-map-2d"]');
  const w = page.locator('.ls .widget[data-widget="linear-map-2d"]');
  await page.waitForSelector('.ls .widget[data-widget="linear-map-2d"][data-ready="true"]');
  assert((await w.locator(".readouts").textContent()).includes("아핀 사상"), "처음에는 아핀");
  await w.locator(".toggle input").uncheck();
  assert((await w.locator(".readouts").textContent()).includes("선형사상"), "bias를 끄면 선형");
  await w.locator(".preset", { hasText: "특이" }).click();
  assert((await w.locator(".readouts").textContent()).includes("det A = 0.00"), "특이 행렬 det");
  await shot("widget-linear-map");
  await closeSheet();
});
await test("relu-regions: 뉴런을 늘리면 영역 수가 는다", async () => {
  await openNode("s:비선형성의-역할");
  await pageTo('.widget[data-widget="relu-regions"]');
  const w = page.locator('.ls .widget[data-widget="relu-regions"]');
  await page.waitForSelector('.ls .widget[data-widget="relu-regions"][data-ready="true"]');
  const count = async () => Number((await w.locator(".readout.hot").textContent()).match(/(\d+)/)[1]);
  await w.locator('input[type="range"]').fill("1");
  const one = await count();
  await w.locator('input[type="range"]').fill("12");
  const many = await count();
  assert(one === 2 && many > 20, `1개: ${one}, 12개: ${many}`);
  await closeSheet();
});
await go(CH123);
await test("loss-to-estimator: 지수분포에서 손실마다 다른 대표값", async () => {
  await openNode("s:결정-이론");
  await pageTo('.widget[data-widget="loss-to-estimator"]');
  const w = page.locator('.ls .widget[data-widget="loss-to-estimator"]');
  await page.waitForSelector('.ls .widget[data-widget="loss-to-estimator"][data-ready="true"]');
  const t = await w.locator(".readouts").textContent();
  const num = (re) => Number(t.match(re)[1]);
  const mean = num(/평균 ([\d.]+)/), med = num(/중앙값 ([\d.]+)/), q90 = num(/90% 분위수 ([\d.]+)/);
  assert(Math.abs(mean - 1) < 0.03 && Math.abs(med - Math.LN2) < 0.03 && Math.abs(q90 - Math.log(10)) < 0.03, t);
  await w.locator(".preset", { hasText: "쌍봉" }).click();
  assert(/평균 -?0\.00/.test(await w.locator(".readouts").textContent()), "쌍봉 평균 0");
  await shot("widget-loss");
  await closeSheet();
});

if (PYO || process.env.PYODIDE_CDN === "1") {
  console.log("브라우저 Python (Pyodide)");
  const runCell = async (cell) => {
    await cell.locator("button.run").click();
    await page.waitForFunction((el) => { const b = el.querySelector("button.run"); const out = el.querySelector(".cell-out"); return b && !b.disabled && out?.dataset.live === "true"; }, await cell.elementHandle(), { timeout: 120000 });
    return cell.locator(".cell-out").textContent();
  };
  await go(CH72);
  await test("코드 셀 실행: 기록된 출력과 같은 결과", async () => {
    await openNode("s:선형-변환과-아핀-변환");
    await pageTo(".cell");
    const out = await runCell(page.locator(".ls .cell:visible").first());
    assert(out.includes("f(0)             = [0.5  0.25]") && !out.includes("Loading"), out);
    await closeSheet();
  });
  await test("같은 session의 뒤 셀만 실행해도 앞 셀을 먼저 실행한다", async () => {
    await page.reload({ waitUntil: "networkidle" });
    await openNode("s:선형층의-합성");
    await pageTo(".cell");
    const out = await runCell(page.locator(".ls .cell:visible").first());
    assert(out.includes("True") && out.includes("rank(W) = 2"), out);
    await closeSheet();
  });
  const LAB = CH72 + "lab/";
  const waitTests = (id) => page.waitForFunction((id) => { const s = document.querySelector(`.tgrp[data-ex="${id}"] .sum`); return s && /\d+\/\d+/.test(s.textContent); }, id, { timeout: 120000 });
  await test("실습실: 창 다섯 개, 시작 코드는 테스트에 실패한다", async () => {
    await go(LAB);
    assert((await page.locator(".win").count()) === 5, "창 수");
    await page.waitForSelector(".editors .cm-content");
    assert((await page.textContent(".editors > div:not([hidden]) .cm-content")).includes("def fold"), "시작 코드");
    await page.click('[data-act="check"]');
    await waitTests("fold-affine-layers");
    assert((await page.locator('.tgrp[data-ex="fold-affine-layers"] li.fail').count()) > 0, "실패가 있어야 함");
    await shot("lab");
  });
  await test("실습실: 해답 코드는 모든 테스트를 통과하고 진도에 기록된다", async () => {
    const sol = await page.evaluate(() => JSON.parse(document.getElementById("page-data").textContent).exercises["relu-breakpoints"].solution);
    await page.evaluate((code) => {
      const k = "roadmap:progress:v1"; const s = JSON.parse(localStorage.getItem(k));
      (s.chapters["linear-algebra/affine-and-nonlinear"].exercises ??= {})["relu-breakpoints"] = { passed: false, code };
      localStorage.setItem(k, JSON.stringify(s));
    }, sol);
    await page.reload({ waitUntil: "networkidle" });
    await page.locator('.w-tabs button[data-k="relu-breakpoints"]').click();
    await page.click('[data-act="check"]');
    await waitTests("relu-breakpoints");
    const g = page.locator('.tgrp[data-ex="relu-breakpoints"]');
    const sum = (await g.locator(".sum").textContent()).trim();
    assert(/^(\d+)\/\1$/.test(sum) && (await g.locator("li.fail").count()) === 0, `${sum}: ${await g.textContent()}`);
    const p = await page.evaluate(() => JSON.parse(localStorage.getItem("roadmap:progress:v1")).chapters["linear-algebra/affine-and-nonlinear"].exercises["relu-breakpoints"]);
    assert(p.passed, JSON.stringify(p));
  });
  await test("실습실 터미널: ls와 python 실행", async () => {
    const tin = page.locator(".term input");
    await tin.fill("ls"); await tin.press("Enter");
    await tin.fill("python cells.py"); await tin.press("Enter");
    await page.waitForFunction(() => document.querySelector(".term").textContent.includes("rank(W) = 2"), null, { timeout: 120000 });
    const t = await page.textContent(".term");
    assert(t.includes("exercises/") && t.includes("f(0)             = [0.5  0.25]"), t.slice(-400));
  });
}

console.log("목차 단계의 장");
// 목차만 있는 장을 빌드 결과에서 하나 찾는다(모든 장이 채워지면 이 검사는 건너뛴다).
const outlineUrl = (() => {
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : e.name === "index.html" ? [path.join(d, e.name)] : []);
  const f = walk(DIST).find((f) => fs.readFileSync(f, "utf8").includes('class="outline-grid'));
  return f ? "/" + path.relative(DIST, path.dirname(f)).split(path.sep).join("/") + "/" : null;
})();
if (outlineUrl) await test("본문이 없는 장은 '다룰 내용' 카드를 한 화면에 보인다", async () => {
  await go(outlineUrl);
  assert((await page.locator(".outline-grid .ocard").count()) >= 5, "카드 수");
  assert((await page.locator(".q-wrap").count()) === 0, "퀘스트 없음");
  await shot("outline");
});

console.log("진도·지도·트랙");
await test("진도 페이지가 기록을 요약한다", async () => {
  await go("/progress/");
  const v = Number(await page.textContent('[data-k="learning"]'));
  const q = Number(await page.textContent('[data-k="quiz"]'));
  assert(v >= 2 && q >= 3, `학습 중 ${v}, 맞힌 퀴즈 ${q}`);
  assert((await page.locator('[data-k="recent"] a').count()) >= 2, "최근 장");
  await shot("progress");
});
await test("로드맵 전체: 트랙을 누르면 연결이 강조되고 서랍이 열린다", async () => {
  await go("/map/");
  assert((await page.locator(".tmap svg.links path").count()) > 10, "트랙 의존선");
  await page.click('.tnode[data-track="probabilistic-inference"]');
  await page.locator(".sheet h2").waitFor();
  assert((await page.textContent(".sheet h2")).includes("확률"), await page.textContent(".sheet h2"));
  assert(await page.locator(".tmap.focusing").count() === 1, "강조 상태");
  await shot("map");
  await page.keyboard.press("Escape");
});
await test("트랙 목록: 레이어 탭을 바꾸면 그 레이어의 트랙만 보인다", async () => {
  await go("/tracks/");
  await page.click('.layer-tabs .tabbar button[data-tab="L2"]');
  const cards = await page.locator('[data-panel="L2"]:not([hidden]) .track-card').allTextContents();
  assert(cards.length === 1 && cards[0].includes("MLC"), cards.join("|"));
  assert(await page.locator('[data-panel="L0"]').isHidden(), "다른 탭은 숨김");
});
await test("트랙 로드맵: 노드 서랍에서 상태를 고르면 노드와 진도가 바뀐다", async () => {
  await go("/tracks/mathematical-foundations/");
  const node = page.locator('[data-node="linear-algebra/affine-and-nonlinear"]');
  assert((await node.getAttribute("data-state")) === "learning", "열어 본 장은 학습 중");
  await node.click();
  await page.locator(".sheet .state-group").waitFor();
  await shot("roadmap-sheet");
  await page.locator(".sheet .state-group button", { hasText: "완료" }).click();
  assert((await node.getAttribute("data-state")) === "done", "완료로 바뀜");
  assert(Number(await page.textContent("[data-track-done]")) === 1, "트랙 완료 수");
  await page.locator(".sheet .state-group button", { hasText: "초기화" }).click();
  await page.keyboard.press("Escape");
  await shot("roadmap");
});
await test("위젯 모음: 구현된 위젯이 모두 뜨고 그림이 그려진다", async () => {
  await go("/widgets/");
  const names = await page.$$eval('section[data-status]:not([data-status="planned"]) .widget', (ns) => ns.map((n) => n.dataset.widget));
  assert(names.length >= 3, `구현된 위젯 ${names.length}개`);
  for (const name of names) {
    const w = page.locator(`.widget[data-widget="${name}"]`);
    await w.scrollIntoViewIfNeeded();
    await page.waitForSelector(`.widget[data-widget="${name}"][data-ready="true"]`, { timeout: 8000 }).catch(() => { throw new Error(`${name} 준비 안 됨`); });
    const box = await w.locator("canvas").first().boundingBox();
    assert(box && box.width > 100 && box.height > 80, `${name} 캔버스 크기 ${JSON.stringify(box)}`);
  }
});
await test("모바일 폭에서 메인·퀘스트·실습실이 가로로 넘치지 않는다", async () => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const u of ["/", CH123, CH72 + "lab/"]) {
    await go(u);
    assert((await over()) <= 0, `${u} 가로 넘침 ${await over()}px`);
  }
  await go(CH123);
  await shot("mobile-quest");
  await openNode("s:학습-목표");
  await shot("mobile-sheet");
});

await browser.close();
server.close();
// 페이지 전환 애니메이션(View Transition)이 빠른 연속 이동으로 끊길 때 Chrome이 남기는 알림은 앱 오류가 아니다.
const real = errors.filter((e) => !/favicon|Transition was aborted because of invalid state/.test(e));
if (real.length) { console.log(`\n콘솔 오류 ${real.length}개:\n  ${real.slice(0, 15).join("\n  ")}`); }
const failed = results.filter(([ok]) => !ok).length;
console.log(`\n${results.length - failed}/${results.length} 통과, 콘솔 오류 ${real.length}개`);
process.exit(failed || real.length ? 1 : 0);
