// 로드맵 전체 (roadmap.sh): 레이어 줄 사이로 트랙 의존선을 그리고, 트랙을 누르면 연결을 강조하고 서랍을 연다.
import { chapterState } from "./store";
import { openSheet, el } from "./sheet";
import type { ChapterSummary } from "../lib/site";
import { coverSVG } from "../lib/cover";

interface T { key: string; code: string; name: string; question: string; outcome: string; layer: number; n: number }

export function mountMap() {
  const data: { tracks: T[]; edges: [string, string][]; chapters: ChapterSummary[] } = JSON.parse(document.getElementById("map-data")!.textContent!);
  const root = document.querySelector<HTMLElement>("[data-tmap]")!;
  const svg = root.querySelector<SVGSVGElement>("svg.links")!;
  const base = document.body.dataset.base || "/";
  const nodes = new Map([...root.querySelectorAll<HTMLButtonElement>("[data-track]")].map((b) => [b.dataset.track!, b]));
  const byKey = new Map(data.tracks.map((t) => [t.key, t]));
  const up = (k: string) => data.edges.filter(([, b]) => b === k).map(([a]) => a);
  const down = (k: string) => data.edges.filter(([a]) => a === k).map(([, b]) => b);
  let focus: string | null = null;

  function draw() {
    const box = root.getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    svg.innerHTML = "";
    for (const [a, b] of data.edges) {
      const na = nodes.get(a), nb = nodes.get(b);
      if (!na || !nb) continue;
      const ra = na.getBoundingClientRect(), rb = nb.getBoundingClientRect();
      const x1 = ra.left + ra.width / 2 - box.left, y1 = ra.bottom - box.top;
      const x2 = rb.left + rb.width / 2 - box.left, y2 = rb.top - box.top;
      if (y2 <= y1) continue; // 같은 줄 안의 관계는 그리지 않는다
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const my = (y1 + y2) / 2;
      path.setAttribute("d", `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`);
      if (focus && (a === focus || b === focus)) path.classList.add("hl");
      svg.append(path);
    }
  }

  function setFocus(k: string | null) {
    focus = k;
    root.classList.toggle("focusing", !!k);
    const hl = new Set(k ? [k, ...up(k), ...down(k)] : []);
    nodes.forEach((b, key) => { b.classList.toggle("hl", hl.has(key)); b.setAttribute("aria-pressed", String(key === k)); });
    draw();
  }

  nodes.forEach((b, key) => b.addEventListener("click", () => {
    setFocus(key);
    const t = byKey.get(key)!;
    const chs = data.chapters.filter((c) => c.track === key);
    const done = chs.filter((c) => chapterState(c.id, { quiz: c.quiz, exercises: c.ex, checks: c.checks }) === "done").length;
    openSheet((panel, close) => {
      const head = el("div", "sheet-head");
      const cv = el("span", "cover wide");
      cv.innerHTML = coverSVG({ id: t.key, track: t.key, code: t.code, ratio: 10 / 16 });
      head.append(cv, el("h2", "", t.name));
      panel.append(head, el("p", "", t.question));
      panel.append(el("h3", "", "도달 목표"), el("p", "muted", t.outcome));
      panel.append(el("h3", "", `진도 ${done} / ${chs.length}`));
      const bar = el("div", "progress"); const i = el("i"); i.style.width = `${chs.length ? (100 * done) / chs.length : 0}%`; bar.append(i); panel.append(bar);
      const link = (keys: string[], title: string) => {
        if (!keys.length) return;
        panel.append(el("h3", "", title));
        const row = el("div", "recent");
        keys.forEach((k2) => {
          const t2 = byKey.get(k2); if (!t2) return;
          const c = el("button", "chip") as HTMLButtonElement;
          c.type = "button";
          c.append(el("span", "n", t2.code), el("span", "", t2.name));
          c.addEventListener("click", () => { close(); nodes.get(k2)?.click(); });
          row.append(c);
        });
        panel.append(row);
      };
      link(up(key), "먼저 필요한 트랙");
      link(down(key), "이 트랙에 기대는 트랙");
      const actions = el("div", "actions");
      const a = el("a", "btn btn-primary btn-lg", "로드맵 열기") as HTMLAnchorElement;
      a.href = `${base}tracks/${key}/`;
      actions.append(a);
      panel.append(actions);
    }, `${t.code} ${t.name}`);
  }));

  new ResizeObserver(draw).observe(root);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setFocus(null); });
  draw();
}
