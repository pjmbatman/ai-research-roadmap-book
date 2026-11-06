// 개념 퀴즈(editorial/conventions.md §9). 단일 선택은 고르는 즉시, 복수 선택·수치는 '확인'으로 채점한다.
// 정답·해설은 공개 데이터다(로그인 없는 오픈소스, AGENTS.md). 채점은 학습을 위한 즉시 피드백이다.
import { progress } from "./store";
import type { PageData } from "./main";

interface Q {
  id: string; type: "single" | "multiple" | "numeric"; objective: number; objectiveText: string;
  prompt: string; explanation: string; choices: { html: string; correct: boolean; feedback: string }[];
  answer?: number; tolerance: number; slider: { min: number; max: number; step: number; unit?: string } | null;
}

const KEYS = "ABCDEFGH";
const h = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, html?: string) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (html !== undefined) el.innerHTML = html;
  return el;
};

export function mountQuizzes(page: PageData) {
  const qs = page.quiz as Q[];
  const inline = new Set<string>();
  document.querySelectorAll<HTMLElement>(".quiz-slot[data-quiz]").forEach((slot) => {
    const q = qs.find((x) => x.id === slot.dataset.quiz);
    if (!q) return;
    inline.add(q.id);
    slot.replaceWith(renderQuestion(page.chapterId, q));
  });
  const all = document.querySelector<HTMLElement>(".quiz-all");
  if (all) renderSummary(all, page, qs, inline);
}

function record(chapterId: string, id: string, correct: boolean) {
  progress.update(chapterId, (p) => {
    const prev = (p.quiz ??= {})[id];
    p.quiz[id] = { correct: correct || !!prev?.correct, tries: (prev?.tries ?? 0) + 1 };
  });
}

function renderQuestion(chapterId: string, q: Q): HTMLElement {
  const box = h("div", "quiz");
  box.id = `quiz-${q.id}`;
  box.dataset.quizId = q.id;
  const head = h("div", "quiz-head");
  const tag = h("span", "tag", q.type === "multiple" ? "개념 확인 · 모두 고르기" : q.type === "numeric" ? "개념 확인 · 값 맞히기" : "개념 확인");
  const obj = h("span", "", `목표 ${q.objective}${q.objectiveText ? ` · ${q.objectiveText}` : ""}`);
  const state = h("span", "quiz-state");
  head.append(tag, obj);
  const prompt = h("div", "quiz-prompt", q.prompt);
  const fb = h("div", "quiz-fb");
  fb.hidden = true;
  fb.setAttribute("aria-live", "polite");
  box.append(head, prompt);

  const prev = progress.get(chapterId).quiz?.[q.id];
  if (prev?.correct) { state.textContent = "✓ 맞힌 문항"; state.className = "quiz-state done"; }
  head.append(state);

  const showResult = (ok: boolean, extra = "") => {
    fb.hidden = false;
    fb.className = `quiz-fb ${ok ? "good" : ""}`;
    fb.innerHTML = ok
      ? `<span class="lbl">정답.</span>${q.explanation}`
      : `<span class="lbl">다시 생각해 보자.</span>${extra}`;
    record(chapterId, q.id, ok);
    if (ok) { state.textContent = "✓ 맞힌 문항"; state.className = "quiz-state done"; }
  };

  if (q.type === "single" || q.type === "multiple") {
    const ul = h("ul", "quiz-choices");
    ul.setAttribute("role", q.type === "single" ? "radiogroup" : "group");
    const buttons: HTMLButtonElement[] = [];
    q.choices.forEach((c, i) => {
      const li = h("li");
      const b = h("button", "quiz-choice") as HTMLButtonElement;
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = `<span class="k">${KEYS[i]}</span><span>${c.html}</span>`;
      b.addEventListener("click", () => {
        if (q.type === "single") {
          buttons.forEach((x) => { x.setAttribute("aria-pressed", "false"); x.classList.remove("is-wrong", "is-right"); });
          b.setAttribute("aria-pressed", "true");
          if (c.correct) { b.classList.add("is-right"); showResult(true); }
          else {
            void b.offsetWidth;
            b.classList.add("is-wrong");
            showResult(false, c.feedback || "이 선택지가 왜 틀렸는지 문항의 조건을 다시 읽어 보자.");
          }
        } else {
          b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
          buttons.forEach((x) => x.classList.remove("is-wrong", "is-right"));
          fb.hidden = true;
        }
      });
      buttons.push(b);
      li.append(b);
      ul.append(li);
    });
    box.append(ul);
    if (q.type === "multiple") {
      const actions = h("div", "quiz-actions");
      const check = h("button", "btn btn-primary", "확인") as HTMLButtonElement;
      check.type = "button";
      const hint = h("small", "", "");
      hint.style.color = "var(--mute)";
      check.addEventListener("click", () => {
        const picked = buttons.map((b) => b.getAttribute("aria-pressed") === "true");
        if (!picked.some(Boolean)) { hint.textContent = "하나 이상 고른 뒤 확인한다."; return; }
        hint.textContent = "";
        const wrong: string[] = [];
        let missed = 0;
        q.choices.forEach((c, i) => {
          if (picked[i] && !c.correct) { buttons[i].classList.add("is-wrong"); wrong.push(`<b>${KEYS[i]}</b> ${c.feedback || "옳지 않은 진술이다."}`); }
          if (picked[i] && c.correct) buttons[i].classList.add("is-right");
          if (!picked[i] && c.correct) missed++;
        });
        const ok = !wrong.length && !missed;
        const parts = [...wrong];
        if (missed) parts.push(`옳은 진술 ${missed}개를 더 골라야 한다.`);
        showResult(ok, parts.join("<br />"));
      });
      actions.append(check, hint);
      box.append(actions);
    }
  } else {
    const wrap = h("div", "quiz-numeric");
    const s = q.slider;
    const unit = s?.unit ?? "";
    const num = h("input") as HTMLInputElement;
    num.type = "number";
    num.setAttribute("aria-label", "답");
    let range: HTMLInputElement | null = null;
    if (s) {
      const row = h("div", "quiz-slider");
      range = h("input") as HTMLInputElement;
      range.type = "range";
      range.min = String(s.min); range.max = String(s.max); range.step = String(s.step);
      range.value = String(s.min);
      range.setAttribute("aria-label", "슬라이더로 답 고르기");
      const out = h("output");
      const sync = (v: string) => { out.textContent = `${v}${unit}`; };
      range.addEventListener("input", () => { num.value = range!.value; sync(range!.value); fb.hidden = true; });
      num.addEventListener("input", () => { if (num.value !== "") { range!.value = num.value; sync(num.value); } });
      sync(range.value);
      num.value = range.value;
      num.min = range.min; num.max = range.max; num.step = range.step;
      row.append(range, out);
      const ticks = h("div", "ticks", `<span>${s.min}${unit}</span><span>${s.max}${unit}</span>`);
      wrap.append(row, ticks);
    }
    const actions = h("div", "quiz-actions");
    const check = h("button", "btn btn-primary", "확인") as HTMLButtonElement;
    check.type = "button";
    const label = h("label", "", "직접 입력 ");
    label.style.fontSize = "14px";
    label.append(num);
    check.addEventListener("click", () => {
      const v = Number(num.value);
      if (num.value === "" || Number.isNaN(v)) { showResult(false, "숫자를 입력한다."); return; }
      const ok = Math.abs(v - (q.answer ?? NaN)) <= (q.tolerance ?? 0) + 1e-12;
      showResult(ok, v < (q.answer ?? 0) ? "그보다 크다." : "그보다 작다.");
    });
    num.addEventListener("keydown", (e) => { if (e.key === "Enter") check.click(); });
    actions.append(label, check);
    wrap.append(actions);
    box.append(wrap);
  }
  box.append(fb);
  return box;
}

function renderSummary(root: HTMLElement, page: PageData, qs: Q[], inline: Set<string>) {
  const card = h("div", "quiz-summary card");
  const update = () => {
    const done = progress.get(page.chapterId).quiz ?? {};
    const rows = page.objectives.map((text, i) => {
      const ofGoal = qs.filter((q) => q.objective === i + 1);
      const ok = ofGoal.filter((q) => done[q.id]?.correct).length;
      const links = ofGoal.map((q, j) => `<a href="#quiz-${q.id}" class="chip ${done[q.id]?.correct ? "done" : ""}">${done[q.id]?.correct ? "✓" : "○"} 문항 ${j + 1}</a>`).join("");
      return `<li><div class="goal"><span class="gn">${i + 1}</span><span>${text}</span><span class="cnt tabular">${ok}/${ofGoal.length}</span></div><div class="links">${links}</div></li>`;
    });
    const total = qs.filter((q) => done[q.id]?.correct).length;
    card.innerHTML = `<p class="sum-head"><b class="tabular">${total} / ${qs.length}</b> 문항을 맞혔다. 학습 목표별로 확인한다.</p><ul class="goal-list">${rows.join("")}</ul>`;
  };
  update();
  window.addEventListener("progress-change", update);
  root.append(card);
  const rest = qs.filter((q) => !inline.has(q.id));
  rest.forEach((q) => root.append(renderQuestion(page.chapterId, q)));
}
