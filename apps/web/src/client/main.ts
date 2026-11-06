// 모든 페이지 공통: 테마, ⌘K 검색, 목차 추적, 진도 표시. 장 페이지에서는 퀴즈·코드 셀·연습문제·위젯을 붙인다.
// 무거운 모듈(CodeMirror, 위젯)은 그 요소가 있는 페이지에서만 동적으로 불러온다.
import { progress, chapterScore, chapterState } from "./store";
import { mountSteps } from "./steps";

const base = document.body.dataset.base || "/";
export const url = (p: string) => base + p;

export interface PageData {
  chapterId: string; title: string; objectives: string[];
  quiz: any[]; exercises: Record<string, any>; widgets: Record<string, { title: string; kind: string; status: string }>;
}
const dataEl = document.getElementById("page-data");
const page: PageData | null = dataEl ? JSON.parse(dataEl.textContent || "null") : null;

/* ---------- 테마 ---------- */
function effectiveTheme(): "dark" | "light" {
  const t = document.documentElement.dataset.theme;
  if (t === "dark" || t === "light") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
document.querySelectorAll<HTMLButtonElement>("[data-theme-toggle]").forEach((b) =>
  b.addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("roadmap:theme", next); } catch {}
    window.dispatchEvent(new CustomEvent("theme-change"));
  }),
);

/* ---------- ⌘K 검색 ---------- */
interface Item { t: string; n: string; e: string; u: string; k: string; h: string }
const palette = document.querySelector<HTMLElement>("[data-palette]");
let index: Item[] | null = null;
let sel = 0;
let lastFocus: Element | null = null;

async function loadIndex() {
  if (!index) index = await fetch(url("search.json")).then((r) => r.json()).catch(() => []);
  return index!;
}
const norm = (s: string) => s.toLowerCase().replace(/\s+/g, "");
function score(it: Item, q: string): number {
  if (!q) return it.k === "트랙" ? 1 : 0;
  const n = norm(q);
  if (norm(it.n) === n) return 100;
  if (norm(it.n).startsWith(n)) return 60;
  if (norm(it.t).includes(n)) return 50 - norm(it.t).indexOf(n) * 0.1;
  if (norm(it.e).includes(n)) return 40;
  if (norm(it.h).includes(n)) return 20;
  return 0;
}
function renderResults(q: string) {
  const ul = palette!.querySelector("ul")!;
  const items = (index || []).map((it) => ({ it, s: score(it, q) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 40);
  sel = Math.min(sel, Math.max(0, items.length - 1));
  ul.innerHTML = "";
  if (!items.length) { ul.innerHTML = `<li class="empty">결과가 없다. 장 번호(예: 12.3)나 영어 제목으로도 찾을 수 있다.</li>`; return; }
  items.forEach(({ it }, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(i === sel));
    const a = document.createElement("a");
    a.href = url(it.u);
    a.innerHTML = `<span class="n"></span><span></span><span class="t"></span>`;
    (a.children[0] as HTMLElement).textContent = it.n;
    (a.children[1] as HTMLElement).textContent = it.t;
    (a.children[2] as HTMLElement).textContent = it.k;
    li.append(a);
    li.addEventListener("mousemove", () => { if (sel !== i) { sel = i; mark(); } });
    ul.append(li);
  });
}
function mark() {
  palette!.querySelectorAll("li[role=option]").forEach((li, i) => li.setAttribute("aria-selected", String(i === sel)));
  palette!.querySelector(`li[aria-selected="true"]`)?.scrollIntoView({ block: "nearest" });
}
async function openPalette() {
  if (!palette) return;
  lastFocus = document.activeElement;
  palette.dataset.open = "true";
  const input = palette.querySelector("input")!;
  input.value = "";
  sel = 0;
  input.focus();
  await loadIndex();
  renderResults(input.value.trim());
}
function closePalette() {
  if (!palette || palette.dataset.open !== "true") return;
  palette.dataset.open = "false";
  (lastFocus as HTMLElement | null)?.focus?.();
}
if (palette) {
  const input = palette.querySelector("input")!;
  input.addEventListener("input", () => { sel = 0; renderResults(input.value.trim()); });
  input.addEventListener("keydown", (e) => {
    const n = palette.querySelectorAll("li[role=option]").length;
    if (e.key === "ArrowDown") { sel = Math.min(n - 1, sel + 1); mark(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); mark(); e.preventDefault(); }
    else if (e.key === "Enter") { palette.querySelector<HTMLAnchorElement>(`li[aria-selected="true"] a`)?.click(); }
  });
  palette.addEventListener("click", (e) => { if (e.target === palette) closePalette(); });
  document.querySelectorAll("[data-open-palette]").forEach((b) => b.addEventListener("click", openPalette));
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); palette.dataset.open === "true" ? closePalette() : openPalette(); }
    else if (e.key === "Escape") closePalette();
    else if (e.key === "/" && !(e.target as HTMLElement).closest("input, textarea, .cm-editor, [contenteditable]")) { e.preventDefault(); openPalette(); }
  });
}

/* ---------- 로딩 막대 ---------- */
// 기다림이 있는 동안 화면 맨 위에 얇은 막대를 보인다(브라우저 Python 불러오기·실행, 페이지 이동).
const bar = document.createElement("div");
bar.className = "loadbar";
bar.setAttribute("aria-hidden", "true");
document.body.append(bar);
const waits = new Set<string>();
export const loading = {
  start(k: string) { waits.add(k); bar.dataset.on = "true"; },
  stop(k: string) { waits.delete(k); if (!waits.size) bar.dataset.on = "false"; },
};
document.addEventListener("click", (e) => {
  const a = (e.target as HTMLElement).closest<HTMLAnchorElement>("a[href]");
  if (!a || e.defaultPrevented || e.metaKey || e.ctrlKey || e.shiftKey || a.target === "_blank") return;
  const u = new URL(a.href, location.href);
  if (u.origin === location.origin && !(u.pathname === location.pathname && u.hash)) loading.start("nav");
});
addEventListener("pageshow", () => loading.stop("nav"));

/* ---------- 방문 표시 ---------- */
function markVisited() {
  const all = progress.all();
  document.querySelectorAll<HTMLElement>("[data-chapter-link]").forEach((a) => {
    a.classList.toggle("visited", !!all[a.dataset.chapterLink!]?.visited);
  });
  trackProgress();
}

/** 트랙 진행: 장 요약(#chapters-data)과 저장된 진도로 트랙별 완료 비율을 계산해 막대와 숫자를 채운다. */
function trackProgress() {
  const el = document.getElementById("chapters-data");
  if (!el) return;
  const chapters: { id: string; track: string; quiz: number; ex: number; checks: number }[] = JSON.parse(el.textContent || "[]");
  const byTrack = new Map<string, { done: number; total: number }>();
  for (const c of chapters) {
    const t = byTrack.get(c.track) ?? { done: 0, total: 0 };
    t.total++;
    if (chapterState(c.id, { quiz: c.quiz, exercises: c.ex, checks: c.checks }) === "done") t.done++;
    byTrack.set(c.track, t);
  }
  document.querySelectorAll<HTMLElement>("[data-track-progress]").forEach((bar) => {
    const t = byTrack.get(bar.dataset.trackProgress!);
    bar.style.width = t && t.total ? `${(100 * t.done) / t.total}%` : "0%";
  });
  document.querySelectorAll<HTMLElement>("[data-track-done]").forEach((n) => { n.textContent = String(byTrack.get(n.dataset.trackDone!)?.done ?? 0); });
}
markVisited();
window.addEventListener("progress-change", markVisited);

/* ---------- 장 페이지 ---------- */
if (page) {
  progress.update(page.chapterId, (p) => { p.visited = Date.now(); });
  mountSteps();
  setupPassChecks(page.chapterId);
  updateMeter(page);
  window.addEventListener("progress-change", () => updateMeter(page));

  if (page.quiz.length) import("./quiz").then((m) => m.mountQuizzes(page));
  if (document.querySelector(".cell")) import("./cells").then((m) => m.mountCells());
  if (document.querySelector(".exercise")) import("./exercise").then((m) => m.mountExercises(page));
  if (document.querySelector(".widget")) import("../widgets/index").then((m) => m.mountWidgets(page));

  if (document.querySelector(".cell[data-exec=browser], .exercise")) {
    import("./runtime").then(({ onRuntimeState }) =>
      onRuntimeState((st) => (st === "loading" || st === "running" ? loading.start("py") : loading.stop("py"))),
    );
  }
}

function setupPassChecks(id: string) {
  const items = document.querySelectorAll<HTMLLIElement>(".box-pass > ul > li");
  items.forEach((li, i) => {
    const key = `c${i + 1}`;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.className = "check";
    box.setAttribute("aria-label", "통과 기준 확인");
    box.checked = !!progress.get(id).checks?.[key];
    box.addEventListener("change", () => progress.update(id, (p) => { (p.checks ??= {})[key] = box.checked; }));
    const span = document.createElement("span");
    while (li.firstChild) span.append(li.firstChild);
    li.append(box, span);
  });
}

function updateMeter(p: PageData) {
  const totals = { quiz: p.quiz.length, exercises: Object.keys(p.exercises).length, checks: document.querySelectorAll(".box-pass > ul > li").length };
  const s = chapterScore(p.chapterId, totals);
  const meter = document.querySelector<HTMLElement>("[data-chapter-meter]");
  if (meter) meter.style.width = `${Math.round(s.ratio * 100)}%`;
  const pct = document.querySelector<HTMLElement>("[data-chapter-pct]");
  if (pct) pct.textContent = `${Math.round(s.ratio * 100)}%`;
}

/* ---------- 탭 바 ---------- */
if (document.querySelector("[data-tabs]")) import("./tabs").then((m) => m.mountAllTabs());

/* ---------- 진도·지도 페이지 ---------- */
if (document.querySelector("[data-progress-page]")) import("./progress-page").then((m) => m.mountProgress());
if (document.querySelector("[data-tmap]")) import("./map").then((m) => m.mountMap());
if (document.querySelector("[data-roadmap]")) import("./roadmap").then((m) => m.mountRoadmap());
