// 홈 대시보드 (ADR 0011): 이어서 학습(최근에 연 장의 커버) · 이번 주 목표 게이지 · 숫자 카드.
import { progress, chapterState } from "./store";
import { coverSVG } from "../lib/cover";
import type { ChapterSummary } from "../lib/site";

const WEEK_GOAL = 5;

export function mountDashboard() {
  const chapters: ChapterSummary[] = JSON.parse(document.getElementById("chapters-data")?.textContent || "[]");
  const base = document.body.dataset.base || "/";
  const $ = (k: string) => document.querySelector<HTMLElement>(`[data-k="${k}"]`);

  const render = () => {
    const all = progress.all();
    const totals = (c: ChapterSummary) => ({ quiz: c.quiz, exercises: c.ex, checks: c.checks });
    const done = chapters.filter((c) => chapterState(c.id, totals(c)) === "done").length;
    const quiz = chapters.reduce((s, c) => s + Object.values(all[c.id]?.quiz ?? {}).filter((x) => x.correct).length, 0);
    const ex = chapters.reduce((s, c) => s + Object.values(all[c.id]?.exercises ?? {}).filter((x) => x.passed).length, 0);
    if ($("done")) $("done")!.textContent = String(done);
    if ($("quiz")) $("quiz")!.textContent = String(quiz);
    if ($("ex")) $("ex")!.textContent = String(ex);

    // 이어서 학습: 최근에 연 장(완료·건너뜀 제외) 네 개
    const recent = chapters
      .filter((c) => all[c.id]?.visited && !["done", "skipped"].includes(chapterState(c.id, totals(c))))
      .sort((a, b) => (all[b.id].visited ?? 0) - (all[a.id].visited ?? 0))
      .slice(0, 4);
    const shelf = document.querySelector<HTMLElement>("[data-continue]");
    if (shelf && recent.length) {
      document.querySelector("[data-continue-title]")!.textContent = "이어서 학습";
      shelf.innerHTML = "";
      for (const c of recent) {
        const a = document.createElement("a");
        a.className = "book";
        a.href = base + c.url;
        const items = c.quiz + c.ex + c.checks;
        const got = Object.values(all[c.id]?.quiz ?? {}).filter((x) => x.correct).length
          + Object.values(all[c.id]?.exercises ?? {}).filter((x) => x.passed).length
          + Object.values(all[c.id]?.checks ?? {}).filter(Boolean).length;
        a.innerHTML = `<span class="cover">${coverSVG({ id: c.id, track: c.track, code: c.code, label: c.label })}</span><b></b><small></small><div class="progress"><i style="width:${items ? (100 * got) / items : 8}%"></i></div>`;
        a.querySelector("b")!.textContent = c.title;
        a.querySelector("small")!.textContent = `${c.label} · ${items ? `${got}/${items}` : "목차"}`;
        shelf.append(a);
      }
    }

    // 이번 주 목표: 지난 7일 동안 연 장
    const weekAgo = Date.now() - 7 * 864e5;
    const week = chapters.filter((c) => (all[c.id]?.visited ?? 0) > weekAgo).length;
    const g = document.querySelector<HTMLElement>("[data-gauge]");
    if (g) {
      g.querySelector("[data-week]")!.textContent = String(week);
      const svg = g.querySelector("svg")!;
      const N = 32, on = Math.round(Math.min(week / WEEK_GOAL, 1) * N);
      let s = "";
      for (let i = 0; i < N; i++) {
        const a = Math.PI - (i / (N - 1)) * Math.PI;
        const x1 = 100 + 74 * Math.cos(a), y1 = 100 - 74 * Math.sin(a);
        const x2 = 100 + 92 * Math.cos(a), y2 = 100 - 92 * Math.sin(a);
        s += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke-width="4" stroke-linecap="round" stroke="${i < on ? "var(--label)" : "var(--track)"}"/>`;
      }
      svg.innerHTML = s;
    }
  };
  render();
  window.addEventListener("progress-change", render);
}
