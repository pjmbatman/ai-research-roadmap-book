// 진도 페이지: 브라우저에 저장된 기록을 벤토 타일로 요약하고, JSON으로 내보내고 가져온다.
import { progress } from "./store";

interface Ch { id: string; label: string; title: string; url: string; status: string; track: string; quiz: number; ex: number; checks: number }

export function mountProgress() {
  const data: { tracks: { key: string; code: string; name: string }[]; chapters: Ch[] } = JSON.parse(document.getElementById("progress-data")!.textContent!);
  const base = document.body.dataset.base || "/";
  const $ = (k: string) => document.querySelector<HTMLElement>(`[data-k="${k}"]`)!;

  const render = () => {
    const all = progress.all();
    const visited = data.chapters.filter((c) => all[c.id]?.visited);
    const count = (f: (p: any) => number) => data.chapters.reduce((s, c) => s + (all[c.id] ? f(all[c.id]) : 0), 0);
    const qOk = count((p) => Object.values(p.quiz ?? {}).filter((x: any) => x.correct).length);
    const eOk = count((p) => Object.values(p.exercises ?? {}).filter((x: any) => x.passed).length);
    const cOk = count((p) => Object.values(p.checks ?? {}).filter(Boolean).length);
    const sum = (k: "quiz" | "ex" | "checks") => data.chapters.reduce((s, c) => s + c[k], 0);
    $("visited").textContent = String(visited.length);
    $("visited-of").textContent = `/${data.chapters.length}`;
    $("quiz").textContent = String(qOk);
    $("quiz-of").textContent = `/${sum("quiz")}`;
    $("ex").textContent = String(eOk);
    $("ex-of").textContent = `/${sum("ex")}`;
    $("checks").textContent = String(cOk);
    $("checks-of").textContent = `/${sum("checks")}`;

    const tr = $("tracks");
    tr.innerHTML = "";
    for (const t of data.tracks) {
      const chs = data.chapters.filter((c) => c.track === t.key);
      if (!chs.length) continue;
      const n = chs.filter((c) => all[c.id]?.visited).length;
      const row = document.createElement("a");
      row.className = "row";
      row.href = `${base}tracks/${t.key}/`;
      row.style.textDecoration = "none";
      row.innerHTML = `<span class="badge badge-code">${t.code}</span><div class="meter"><i style="width:${(100 * n) / chs.length}%"></i></div><span class="tabular" style="text-align:right">${n}/${chs.length}</span>`;
      row.title = t.name;
      tr.append(row);
    }
    const rec = $("recent");
    rec.innerHTML = "";
    const recent = visited.sort((a, b) => (all[b.id].visited ?? 0) - (all[a.id].visited ?? 0)).slice(0, 8);
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
    msg.textContent = "✓";
  });
  document.querySelector<HTMLInputElement>("[data-act=import]")!.addEventListener("change", async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try { progress.import(await file.text()); msg.textContent = "✓"; } catch (err: any) { msg.textContent = err.message; }
  });
  document.querySelector("[data-act=reset]")!.addEventListener("click", () => {
    if (confirm("이 브라우저에 저장된 진도를 모두 지울까? 되돌릴 수 없다.")) { progress.reset(); msg.textContent = "✓"; }
  });
}
