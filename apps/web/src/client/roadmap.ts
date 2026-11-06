// 트랙 로드맵 (roadmap.sh): 노드 모양이 장 상태를 보이고, 노드를 누르면 서랍에서 상태를 직접 고르거나 장을 연다.
import { progress, chapterState, STATE_LABEL } from "./store";
import { openSheet, el } from "./sheet";
import type { ChapterSummary } from "../lib/site";
import { coverSVG } from "../lib/cover";

export function mountRoadmap() {
  const chapters: ChapterSummary[] = JSON.parse(document.getElementById("chapters-data")!.textContent || "[]");
  const byId = new Map(chapters.map((c) => [c.id, c]));
  const base = document.body.dataset.base || "/";
  const nodes = [...document.querySelectorAll<HTMLButtonElement>("[data-node]")];
  const stateOf = (c: ChapterSummary) => chapterState(c.id, { quiz: c.quiz, exercises: c.ex, checks: c.checks });

  const paint = () => nodes.forEach((n) => {
    const c = byId.get(n.dataset.node!);
    if (!c) return;
    n.dataset.state = stateOf(c);
    if (c.ready && !n.querySelector(".ready")) n.prepend(Object.assign(document.createElement("span"), { className: "ready", title: "본문 있음" }));
  });
  paint();
  window.addEventListener("progress-change", paint);

  nodes.forEach((n) => n.addEventListener("click", () => {
    const c = byId.get(n.dataset.node!);
    if (!c) return;
    openSheet((panel) => {
      const head = el("div", "sheet-head");
      const st = el("span", "badge");
      const cv = el("span", "cover");
      cv.innerHTML = coverSVG({ id: c.id, track: c.track, code: c.code, label: c.label });
      head.append(cv, el("h2", "", c.title), st);
      panel.append(head);
      if (c.goals.length) {
        panel.append(el("h3", "", "이 장을 마치면"));
        const ul = el("ul");
        c.goals.forEach((g) => ul.append(el("li", "", g)));
        panel.append(ul);
      }
      const pres = c.pre.map((id) => byId.get(id)).filter(Boolean) as ChapterSummary[];
      if (pres.length) {
        panel.append(el("h3", "", "먼저 읽기"));
        const row = el("div", "recent");
        pres.forEach((p) => {
          const a = el("a", "chip");
          a.href = base + p.url;
          a.append(el("span", "n", p.label), el("span", "", p.title));
          row.append(a);
        });
        panel.append(row);
      }
      panel.append(el("h3", "", "상태"));
      const group = el("div", "state-group");
      const opts: ["learning" | "done" | "skipped" | "reset", string][] = [["learning", "학습 중"], ["done", "완료"], ["skipped", "건너뜀"], ["reset", "초기화"]];
      const buttons = opts.map(([k, label]) => {
        const b = el("button", "btn btn-sm", label) as HTMLButtonElement;
        b.type = "button";
        b.addEventListener("click", () => {
          progress.update(c.id, (p) => { if (k === "reset") { delete p.mark; delete p.visited; } else p.mark = k; });
          sync();
        });
        group.append(b);
        return [k, b] as const;
      });
      panel.append(group);
      const actions = el("div", "actions");
      const open = el("a", "btn btn-primary btn-lg") as HTMLAnchorElement;
      open.href = base + c.url;
      actions.append(open);
      panel.append(actions);
      const sync = () => {
        const s = stateOf(c);
        st.textContent = STATE_LABEL[s];
        const mark = progress.get(c.id).mark;
        buttons.forEach(([k, b]) => b.setAttribute("aria-pressed", String(k === mark)));
        open.textContent = !c.ready ? "다룰 내용 보기" : s === "none" ? "읽기 시작" : "이어서 읽기";
      };
      sync();
    }, `${c.label} ${c.title}`);
  }));
}
