// 내 진도: 장 상태(roadmap.sh 모델)를 집계해 숫자 카드·트랙 막대·최근 목록으로 보이고, JSON으로 내보내고 가져온다.
import { progress, chapterState } from "./store";
import type { ChapterSummary } from "../lib/site";

export function mountProgress() {
  const data: { tracks: { key: string; code: string; name: string }[]; chapters: ChapterSummary[] } = JSON.parse(document.getElementById("progress-data")!.textContent!);
  const base = document.body.dataset.base || "/";
  const $ = (k: string) => document.querySelector<HTMLElement>(`[data-k="${k}"]`)!;
  const stateOf = (c: ChapterSummary) => chapterState(c.id, { quiz: c.quiz, exercises: c.ex, checks: c.checks });

  const render = () => {
    const all = progress.all();
    const states = new Map(data.chapters.map((c) => [c.id, stateOf(c)]));
    const n = (s: string) => [...states.values()].filter((x) => x === s).length;
    const count = (f: (p: any) => number) => data.chapters.reduce((s, c) => s + (all[c.id] ? f(all[c.id]) : 0), 0);
    const sum = (k: "quiz" | "ex") => data.chapters.reduce((s, c) => s + c[k], 0);
    $("done").textContent = String(n("done"));
    $("done-of").textContent = `/${data.chapters.length}`;
    $("learning").textContent = String(n("learning"));
    $("quiz").textContent = String(count((p) => Object.values(p.quiz ?? {}).filter((x: any) => x.correct).length));
    $("quiz-of").textContent = `/${sum("quiz")}`;
    $("ex").textContent = String(count((p) => Object.values(p.exercises ?? {}).filter((x: any) => x.passed).length));
    $("ex-of").textContent = `/${sum("ex")}`;

    const tr = $("tracks");
    tr.innerHTML = "";
    for (const t of data.tracks) {
      const chs = data.chapters.filter((c) => c.track === t.key);
      if (!chs.length) continue;
      const d = chs.filter((c) => states.get(c.id) === "done").length;
      const a = document.createElement("a");
      a.href = `${base}tracks/${t.key}/`;
      a.title = t.name;
      a.innerHTML = `<span class="badge badge-code">${t.code}</span><div class="progress"><i style="width:${(100 * d) / chs.length}%"></i></div><span class="muted tabular" style="text-align:right">${d}/${chs.length}</span>`;
      tr.append(a);
    }
    const rec = $("recent");
    rec.innerHTML = "";
    const recent = data.chapters.filter((c) => all[c.id]?.visited).sort((a, b) => (all[b.id].visited ?? 0) - (all[a.id].visited ?? 0)).slice(0, 8);
    if (!recent.length) rec.innerHTML = `<a class="btn btn-primary" href="${base}orientation/how-to-read/">시작하기</a>`;
    for (const c of recent) {
      const a = document.createElement("a");
      a.className = "chip";
      a.href = base + c.url;
      a.innerHTML = `<span class="n"></span><span></span>`;
      (a.children[0] as HTMLElement).textContent = c.label;
      (a.children[1] as HTMLElement).textContent = c.title;
      rec.append(a);
    }
  };
  render();
  window.addEventListener("progress-change", render);

  const msg = $("msg");
  document.querySelector("[data-act=export]")!.addEventListener("click", () => {
    const blob = new Blob([progress.export()], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ai-for-science-roadmap-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    msg.textContent = "내보냈다";
  });
  document.querySelector<HTMLInputElement>("[data-act=import]")!.addEventListener("change", async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try { progress.import(await file.text()); msg.textContent = "가져왔다"; } catch (err: any) { msg.textContent = err.message; }
  });
  document.querySelector("[data-act=reset]")!.addEventListener("click", () => {
    if (confirm("이 브라우저에 저장된 진도를 모두 지울까? 되돌릴 수 없다.")) { progress.reset(); msg.textContent = "지웠다"; }
  });
}
