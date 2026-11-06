// 연습문제(editorial/conventions.md §8). 학습자의 코드를 submission.py로 두고, 저장소의 test_exercise.py를
// 브라우저 Python에서 그대로 실행해 채점한다(platform-architecture.md §2.4). 작성 중인 코드는 브라우저에 저장한다.
import { createEditor } from "./editor";
import { session } from "./runtime";
import { HARNESS } from "./harness";
import { progress } from "./store";
import { cleanError } from "./cells";
import { tabs } from "./tabs";
import type { PageData } from "./main";

export function mountExercises(page: PageData) {
  groupSiblings(page);
  document.querySelectorAll<HTMLElement>(".exercise[data-exercise]").forEach((root) => {
    const ex = page.exercises[root.dataset.exercise!];
    if (!ex) return;
    const saved = progress.get(page.chapterId).exercises?.[ex.id];
    root.innerHTML = `
      <div class="ex-head"><h3></h3><span class="ex-state"></span></div>
      <div data-tabs class="ex-tabs">
        <div class="tabbar"><button data-tab="problem">문제</button><button data-tab="code">코드</button><button data-tab="solution">해답</button></div>
        <div data-panel="problem" class="ex-body"></div>
        <div data-panel="code">
          <div class="ex-editor"></div>
          <div class="ex-bar">
            <button type="button" class="btn btn-primary" data-run>▶ 테스트</button>
            <button type="button" class="btn" data-reset>처음 코드로</button>
            <span class="cell-status" aria-live="polite"></span>
          </div>
          <ul class="ex-results" aria-live="polite"></ul>
        </div>
        <div data-panel="solution" class="ex-solution"><pre><code></code></pre></div>
      </div>`;
    root.querySelector("h3")!.textContent = ex.titleText;
    tabs(root.querySelector<HTMLElement>("[data-tabs]")!);
    root.querySelector(".ex-body")!.innerHTML = ex.problem;
    root.querySelector(".ex-solution code")!.textContent = ex.solution;
    const state = root.querySelector<HTMLElement>(".ex-state")!;
    const status = root.querySelector<HTMLElement>(".cell-status")!;
    const results = root.querySelector<HTMLElement>(".ex-results")!;
    const runBtn = root.querySelector<HTMLButtonElement>("[data-run]")!;
    const setState = (passed?: boolean) => {
      state.textContent = passed ? "✓ 통과" : "";
      state.classList.toggle("ok", !!passed);
    };
    setState(saved?.passed ?? undefined);

    let timer: number | undefined;
    const ed = createEditor(root.querySelector(".ex-editor")!, saved?.code ?? ex.starter, {
      label: `${ex.titleText} 코드 편집기`,
      onRun: () => runBtn.click(),
      onChange: (code) => {
        clearTimeout(timer);
        timer = window.setTimeout(() => progress.update(page.chapterId, (p) => {
          const prev = (p.exercises ??= {})[ex.id];
          p.exercises[ex.id] = { passed: prev?.passed ?? false, code };
        }), 500);
      },
    });

    root.querySelector("[data-reset]")!.addEventListener("click", () => {
      if (ed.get() !== ex.starter && !confirm("작성한 코드를 지우고 처음 코드로 되돌릴까?")) return;
      ed.set(ex.starter);
    });

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      status.textContent = "";
      runBtn.textContent = "채점 중";
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
      for (const t of list) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="${t.ok ? "ok" : "fail"}">${t.ok ? "✓" : "✗"}</span><span class="nm"></span>${t.ok ? "" : `<span class="msg"></span>`}`;
        li.querySelector(".nm")!.textContent = t.name;
        if (!t.ok) li.querySelector(".msg")!.textContent = t.msg ?? "";
        results.append(li);
      }
      if (r.stdout.trim()) {
        const li = document.createElement("li");
        li.innerHTML = `<span>›</span><span>출력</span><span class="msg"></span>`;
        li.querySelector(".msg")!.textContent = r.stdout.trim();
        results.append(li);
      }
      setState(passed);
      progress.update(page.chapterId, (p) => {
        const prev = (p.exercises ??= {})[ex.id];
        p.exercises[ex.id] = { passed: passed || !!prev?.passed, code: ed.get() };
      });
    });
  });
}

/** 같은 자리에 이어진 연습문제 여러 개를 탭 하나로 묶는다. */
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
      b.textContent = page.exercises[x.dataset.exercise!]?.titleText ?? `연습 ${i + 1}`;
      bar.append(b);
      x.dataset.panel = `ex${i}`;
      box.append(x);
    });
    tabs(box);
  });
}
