// 메인(Bento): 저장된 진도로 타일을 채운다. 이어서 학습 타일은 최근에 연 '본문 있는 장'(없으면 첫 장)의 퀘스트 진행을 점으로 보인다.
import { progress, chapterState } from "./store";
import { questStates } from "./quest-state";
import { TRACK_COLORS } from "../lib/cover";
import type { ChapterSummary } from "../lib/site";

const WEEK_GOAL = 5;

export function mountDashboard() {
  const chapters: ChapterSummary[] = JSON.parse(document.getElementById("chapters-data")?.textContent || "[]");
  const base = document.body.dataset.base || "/";
  const $ = (k: string) => document.querySelector<HTMLElement>(`[data-k="${k}"]`);
  const totals = (c: ChapterSummary) => ({ quiz: c.quiz, exercises: c.ex, checks: c.checks });
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // 전체 장 수: 0에서 세어 올라간다
  const cnt = document.querySelector<HTMLElement>("[data-count]");
  if (cnt && !reduced) {
    const n = Number(cnt.dataset.count), t0 = performance.now();
    const f = (t: number) => { const p = Math.min(1, (t - t0) / 1100); cnt.textContent = String(Math.round(n * (1 - Math.pow(1 - p, 3)))); if (p < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  }

  const render = () => {
    const all = progress.all();
    const state = (c: ChapterSummary) => chapterState(c.id, totals(c));

    // 이어서 학습
    const ready = chapters.filter((c) => c.ready && c.nodes.length);
    const recent = ready.filter((c) => all[c.id]?.visited && state(c) !== "done" && state(c) !== "skipped").sort((a, b) => (all[b.id].visited ?? 0) - (all[a.id].visited ?? 0));
    const pick = recent[0] ?? ready.find((c) => !c.id.startsWith("appendices/") && state(c) !== "done") ?? ready[0];
    const hero = document.querySelector<HTMLAnchorElement>("[data-hero]");
    if (hero && pick) {
      const h = (k: string) => hero.querySelector<HTMLElement>(`[data-h="${k}"]`)!;
      const [c1, c2] = TRACK_COLORS[pick.track] ?? TRACK_COLORS.reference;
      hero.href = base + pick.url;
      hero.style.setProperty("--t1", c1);
      hero.style.setProperty("--t2", c2);
      h("crumb").textContent = pick.crumb;
      h("label").textContent = pick.label;
      h("title").textContent = pick.title;
      h("lead").innerHTML = pick.lead;
      const st = questStates(pick.id, pick.nodes, pick.exIds);
      h("path").innerHTML = pick.nodes.map((_, i) => `${i ? "<span></span>" : ""}<i class="${st.done[i] ? "done" : i === st.cur ? "cur" : ""}"></i>`).join("");
      h("go").textContent = st.count === 0 ? "퀘스트 시작" : st.cur < 0 ? "다시 보기" : "퀘스트 계속";
      h("next").textContent = st.cur < 0 ? `모두 마쳤다 · ${pick.nodes.length}단계` : `다음: ${pick.nodes[st.cur].name} · ${st.count}/${pick.nodes.length} 단계`;
    }

    // 이번 주
    const weekAgo = Date.now() - 7 * 864e5;
    const week = chapters.filter((c) => (all[c.id]?.visited ?? 0) > weekAgo).length;
    if ($("week")) $("week")!.textContent = String(week);
    const ring = $("ring");
    if (ring) { const C = 2 * Math.PI * 52; ring.style.opacity = week ? "1" : "0"; requestAnimationFrame(() => ring.setAttribute("stroke-dasharray", `${(Math.min(week, WEEK_GOAL) / WEEK_GOAL) * C} ${C}`)); }

    // 본문이 있는 장의 상태 점
    document.querySelectorAll<HTMLElement>("[data-state-of]").forEach((d) => {
      const c = chapters.find((x) => x.id === d.dataset.stateOf);
      d.className = `state-dot ${c ? state(c) : ""}`;
      d.title = c ? { none: "시작 전", learning: "학습 중", done: "완료", skipped: "건너뜀" }[state(c)] : "";
    });

    // 실습실 타일: 연습문제별 통과 여부를 터미널처럼 찍는다
    const pre = $("labpre");
    if (pre) {
      const ids: string[] = JSON.parse(pre.dataset.ex || "[]");
      const p = all[pre.dataset.ch!] ?? {};
      const w = Math.max(...ids.map((x) => x.length)) + 3;
      const lines = [`<span class="p">learner@ai4s-lab</span>:~/lab$ check`, ...ids.map((id) => `${id.padEnd(w)}${p.exercises?.[id]?.passed ? '<span class="ok">✓ 통과</span>' : '<span class="e">✗ 아직</span>'}`), "", `<span class="p">learner@ai4s-lab</span>:~/lab$ <span class="cur"></span>`];
      if (reduced) pre.innerHTML = lines.join("\n");
      else { let i = 0; pre.innerHTML = ""; const step = () => { pre.innerHTML = lines.slice(0, ++i).join("\n"); if (i < lines.length) setTimeout(step, 220); }; setTimeout(step, 500); }
    }

    // 트랙 진행
    document.querySelectorAll<HTMLElement>("[data-heat]").forEach((a) => {
      const chs = chapters.filter((c) => c.track === a.dataset.heat);
      const d = chs.filter((c) => state(c) === "done").length + 0.5 * chs.filter((c) => state(c) === "learning").length;
      const pct = chs.length ? d / chs.length : 0;
      a.style.setProperty("--a", `${Math.round(12 + pct * 70)}%`);
      a.querySelector("[data-heat-pct]")!.textContent = `${Math.round(pct * 100)}%`;
    });

    // 숫자
    const done = chapters.filter((c) => state(c) === "done").length;
    const quiz = chapters.reduce((s, c) => s + Object.values(all[c.id]?.quiz ?? {}).filter((x) => x.correct).length, 0);
    const ex = chapters.reduce((s, c) => s + Object.values(all[c.id]?.exercises ?? {}).filter((x) => x.passed).length, 0);
    if ($("done")) $("done")!.textContent = String(done);
    if ($("quiz")) $("quiz")!.textContent = String(quiz);
    if ($("ex")) $("ex")!.textContent = String(ex);
  };
  render();
  window.addEventListener("progress-change", render);
}
