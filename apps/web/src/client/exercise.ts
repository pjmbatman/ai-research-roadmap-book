// 연습문제 (editorial/benchmarks.md §3.2). 학습자의 코드를 submission.py로 두고, 저장소의 test_exercise.py를
// 브라우저 Python에서 그대로 실행해 채점한다(platform-architecture.md §2.4).
// 배치: 넓은 화면은 왼쪽 문제 | 오른쪽 편집기·테스트 결과(svelte·TutorialKit), 좁은 화면은 '문제 ⇄ 코드' 전환.
// 편집기 머리의 버튼 하나가 '해답 보기 ⇄ 내 코드로'를 오간다(TutorialKit Solve⇄Reset). 테스트를 모두 통과하면 자동 완료(svelte).
import { createEditor } from "./editor";
import { session } from "./runtime";
import { HARNESS } from "./harness";
import { progress } from "./store";
import { cleanError } from "./cells";
import { tabs } from "./tabs";
import type { PageData } from "./main";

const RESET_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>';

export function mountExercises(page: PageData) {
  groupSiblings(page);
  document.querySelectorAll<HTMLElement>(".exercise[data-exercise]").forEach((root) => {
    const ex = page.exercises[root.dataset.exercise!];
    if (!ex) return;
    const saved = progress.get(page.chapterId).exercises?.[ex.id];
    root.innerHTML = `
      <div class="ex-head"><h3></h3><span class="ex-state"></span>
        <div class="ex-switch" role="tablist"><button type="button" data-view="problem" aria-selected="true">문제</button><button type="button" data-view="code" aria-selected="false">코드</button></div>
      </div>
      <div class="ex-split">
        <div class="ex-body"></div>
        <div class="ex-work">
          <div class="ex-edhead">
            <span class="fname">submission.py</span>
            <span class="spacer"></span>
            <button type="button" class="btn btn-sm" data-solve>해답 보기</button>
            <button type="button" class="icon-btn" data-reset aria-label="처음 코드로" title="처음 코드로">${RESET_ICON}</button>
          </div>
          <div class="ex-editor"></div>
          <div class="ex-bar">
            <button type="button" class="btn btn-primary" data-run>▶ 테스트</button>
            <span class="cell-status" aria-live="polite"></span>
            <span class="spacer"></span>
            <span data-next-slot></span>
          </div>
          <ul class="ex-results" aria-live="polite"></ul>
        </div>
      </div>`;
    root.querySelector("h3")!.textContent = ex.titleText;
    root.querySelector(".ex-body")!.innerHTML = ex.problem;
    const state = root.querySelector<HTMLElement>(".ex-state")!;
    const status = root.querySelector<HTMLElement>(".cell-status")!;
    const results = root.querySelector<HTMLElement>(".ex-results")!;
    const runBtn = root.querySelector<HTMLButtonElement>("[data-run]")!;
    const solveBtn = root.querySelector<HTMLButtonElement>("[data-solve]")!;
    const setState = (passed?: boolean) => {
      state.textContent = passed ? "✓ 통과" : "";
      state.classList.toggle("ok", !!passed);
      root.classList.toggle("passed", !!passed);
    };
    setState(saved?.passed);

    // 좁은 화면의 '문제 ⇄ 코드' 전환
    const views = root.querySelectorAll<HTMLButtonElement>(".ex-switch button");
    const view = (v: string) => {
      root.dataset.view = v;
      views.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === v)));
    };
    views.forEach((b) => b.addEventListener("click", () => view(b.dataset.view!)));
    view("problem");

    let timer: number | undefined;
    let showingSolution = false;
    let mine = "";
    const ed = createEditor(root.querySelector(".ex-editor")!, saved?.code ?? ex.starter, {
      label: `${ex.titleText} 코드 편집기`,
      onRun: () => runBtn.click(),
      onChange: (code) => {
        if (showingSolution) return;
        clearTimeout(timer);
        timer = window.setTimeout(() => progress.update(page.chapterId, (p) => {
          const prev = (p.exercises ??= {})[ex.id];
          p.exercises[ex.id] = { passed: prev?.passed ?? false, code };
        }), 500);
      },
    });

    solveBtn.disabled = !ex.solution;
    solveBtn.addEventListener("click", () => {
      if (!showingSolution) { mine = ed.get(); showingSolution = true; ed.set(ex.solution); solveBtn.textContent = "내 코드로"; }
      else { showingSolution = false; ed.set(mine); solveBtn.textContent = "해답 보기"; }
    });
    root.querySelector("[data-reset]")!.addEventListener("click", () => {
      if (ed.get() !== ex.starter && !confirm("작성한 코드를 지우고 처음 코드로 되돌릴까?")) return;
      showingSolution = false;
      solveBtn.textContent = "해답 보기";
      ed.set(ex.starter);
    });

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      runBtn.textContent = "채점 중";
      status.textContent = "";
      results.innerHTML = "";
      const dir = `/home/pyodide/exercises/${ex.id}`;
      const r = await session(`exercise:${ex.id}`).run(`${HARNESS}\nrun_exercise_tests(${JSON.stringify(dir)})`, {
        files: { [`${dir}/submission.py`]: ed.get(), [`${dir}/test_exercise.py`]: ex.tests },
      });
      runBtn.disabled = false;
      runBtn.textContent = "▶ 테스트";
      if (!r.ok) {
        status.textContent = "오류";
        results.innerHTML = `<li><span class="fail">✗</span><span>채점기를 실행하지 못했다.</span><span class="msg"></span></li>`;
        results.querySelector(".msg")!.textContent = cleanError(r.error || "");
        return;
      }
      const list: { name: string; ok: boolean; msg?: string }[] = JSON.parse(String(r.value));
      const passed = list.length > 0 && list.every((t) => t.ok);
      status.textContent = `${list.filter((t) => t.ok).length} / ${list.length} 통과`;
      status.classList.toggle("all", passed);
      for (const t of list) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="${t.ok ? "ok" : "fail"}">${t.ok ? "✓" : "✗"}</span><span class="nm"></span>${t.ok ? "" : `<span class="msg"></span>`}`;
        li.querySelector(".nm")!.textContent = t.name.replace(/^test_/, "").replace(/_/g, " ");
        if (!t.ok) li.querySelector(".msg")!.textContent = t.msg ?? "";
        results.append(li);
      }
      if (r.stdout.trim()) {
        const li = document.createElement("li");
        li.innerHTML = `<span>›</span><span class="msg"></span>`;
        li.querySelector(".msg")!.textContent = r.stdout.trim();
        results.append(li);
      }
      // 해답을 띄워 놓고 통과한 것은 완료로 치지 않는다.
      const earned = passed && !showingSolution;
      setState(earned || saved?.passed);
      progress.update(page.chapterId, (p) => {
        const prev = (p.exercises ??= {})[ex.id];
        p.exercises[ex.id] = { passed: earned || !!prev?.passed, code: showingSolution ? (prev?.code ?? mine) : ed.get() };
      });
      if (passed) root.dispatchEvent(new CustomEvent("exercise-passed", { bubbles: true }));
    });
  });
}

/** 같은 자리에 이어진 연습문제 여러 개를 'k / N · 제목' 탭 하나로 묶고, 통과하면 '다음 연습' 버튼을 보인다. */
function groupSiblings(page: PageData) {
  const seen = new Set<Element>();
  document.querySelectorAll<HTMLElement>(".exercise[data-exercise]").forEach((el) => {
    if (seen.has(el)) return;
    const run: HTMLElement[] = [el];
    let n = el.nextElementSibling;
    while (n && n.matches(".exercise[data-exercise]")) { run.push(n as HTMLElement); n = n.nextElementSibling; }
    run.forEach((x) => seen.add(x));
    if (run.length < 2) return;
    const box = document.createElement("div");
    box.className = "ex-group";
    box.dataset.tabs = "";
    const bar = document.createElement("div");
    bar.className = "tabbar";
    el.before(box);
    box.append(bar);
    run.forEach((x, i) => {
      const b = document.createElement("button");
      b.dataset.tab = `ex${i}`;
      b.innerHTML = `<span class="k tabular">${i + 1} / ${run.length}</span><span></span>`;
      (b.lastChild as HTMLElement).textContent = page.exercises[x.dataset.exercise!]?.titleText ?? "";
      bar.append(b);
      x.dataset.panel = `ex${i}`;
      box.append(x);
    });
    const t = tabs(box);
    box.addEventListener("exercise-passed", (e) => {
      const ex = (e.target as HTMLElement).closest<HTMLElement>(".exercise")!;
      const i = run.indexOf(ex);
      const slot = ex.querySelector<HTMLElement>("[data-next-slot]");
      if (i < 0 || i === run.length - 1 || !slot || slot.firstChild) return;
      const nb = document.createElement("button");
      nb.type = "button";
      nb.className = "btn";
      nb.textContent = "다음 연습 →";
      nb.addEventListener("click", () => { t.select(`ex${i + 1}`); box.scrollIntoView({ block: "start", behavior: "smooth" }); });
      slot.append(nb);
    });
  });
}
