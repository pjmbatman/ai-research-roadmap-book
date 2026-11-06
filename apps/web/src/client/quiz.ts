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
  // 본문 속 문항: 버튼 하나. 누르면 문항 카드를 대화상자로 띄운다.
  document.querySelectorAll<HTMLElement>(".quiz-slot[data-quiz]").forEach((slot) => {
    const q = qs.find((x) => x.id === slot.dataset.quiz);
    if (!q) return;
    const b = h("button", "quiz-open") as HTMLButtonElement;
    b.type = "button";
    b.dataset.quizOpen = q.id;
    const sync = () => {
      const ok = !!progress.get(page.chapterId).quiz?.[q.id]?.correct;
      b.classList.toggle("done", ok);
      b.innerHTML = `<span class="dot"></span>${ok ? "확인 문제 · 맞힘" : "확인 문제"}`;
    };
    sync();
    window.addEventListener("progress-change", sync);
    b.addEventListener("click", () => openDialog(page.chapterId, q));
    slot.replaceWith(b);
  });
  const all = document.querySelector<HTMLElement>(".quiz-all");
  if (all) mountCarousel(all, page, qs);
}

let dialog: HTMLDialogElement | null = null;
function openDialog(chapterId: string, q: Q) {
  if (!dialog) {
    dialog = h("dialog", "quiz-dialog") as HTMLDialogElement;
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog!.close(); });
    document.body.append(dialog);
  }
  dialog.innerHTML = "";
  const close = h("button", "dialog-close", "×") as HTMLButtonElement;
  close.type = "button";
  close.setAttribute("aria-label", "닫기");
  close.addEventListener("click", () => dialog!.close());
  dialog.append(close, renderQuestion(chapterId, q));
  dialog.showModal();
}

/** 개념 퀴즈 단계: 한 번에 한 문항. 위쪽 막대가 진행과 정답을 보여 준다. 마지막은 학습 목표별 요약. */
function mountCarousel(root: HTMLElement, page: PageData, qs: Q[]) {
  const bar = h("div", "qbar");
  const stage = h("div", "qstage");
  const nav = h("div", "qnav");
  const prev = h("button", "btn", "←") as HTMLButtonElement;
  const next = h("button", "btn btn-primary", "다음 문항") as HTMLButtonElement;
  prev.type = next.type = "button";
  nav.append(prev, next);
  root.append(bar, stage, nav);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let cur = -1;
  const done = () => progress.get(page.chapterId).quiz ?? {};
  const segs = [...qs, null].map((q, i) => {
    const b = h("button", "seg") as HTMLButtonElement;
    b.type = "button";
    b.setAttribute("aria-label", q ? `문항 ${i + 1}` : "요약");
    if (!q) b.classList.add("seg-sum");
    b.addEventListener("click", () => go(i));
    bar.append(b);
    return b;
  });
  const paint = () => segs.forEach((b, i) => {
    b.setAttribute("aria-current", String(i === cur));
    b.classList.toggle("ok", i < qs.length && !!done()[qs[i].id]?.correct);
  });
  function go(i: number) {
    i = Math.max(0, Math.min(qs.length, i));
    const dir = i >= cur ? 1 : -1;
    cur = i;
    stage.innerHTML = "";
    const card = i < qs.length ? renderQuestion(page.chapterId, qs[i]) : summary(page, qs, go);
    if (!reduced) card.classList.add(dir > 0 ? "enter-next" : "enter-prev");
    stage.append(card);
    prev.disabled = i === 0;
    next.hidden = i === qs.length;
    next.textContent = i === qs.length - 1 ? "요약 보기" : "다음 문항";
    paint();
  }
  prev.addEventListener("click", () => go(cur - 1));
  next.addEventListener("click", () => go(cur + 1));
  window.addEventListener("progress-change", paint);
  const first = qs.findIndex((q) => !done()[q.id]?.correct);
  go(first < 0 ? qs.length : first);
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
  const tag = h("span", "tag", q.type === "multiple" ? "모두 고르기" : "");
  const state = h("span", "quiz-state");
  head.append(tag);
  const prompt = h("div", "quiz-prompt", q.prompt);
  const fb = h("div", "quiz-fb");
  fb.hidden = true;
  fb.setAttribute("aria-live", "polite");
  box.append(head, prompt);

  const prev = progress.get(chapterId).quiz?.[q.id];
  if (prev?.correct) { box.classList.add("solved"); }
  head.append(state);

  const showResult = (ok: boolean, extra = "") => {
    fb.hidden = false;
    fb.className = `quiz-fb ${ok ? "good" : ""}`;
    fb.innerHTML = ok
      ? `<span class="lbl">정답.</span>${q.explanation}`
      : `<span class="lbl">다시 생각해 보자.</span>${extra}`;
    record(chapterId, q.id, ok);
    if (ok) box.classList.add("solved");
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
      const hint = h("span", "quiz-hint", "");
      check.addEventListener("click", () => {
        const picked = buttons.map((b) => b.getAttribute("aria-pressed") === "true");
        if (!picked.some(Boolean)) { hint.textContent = "하나 이상 고른다"; return; }
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
      wrap.append(row);
    }
    const actions = h("div", "quiz-actions");
    const check = h("button", "btn btn-primary", "확인") as HTMLButtonElement;
    check.type = "button";
    const label = h("label", "");
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

function summary(page: PageData, qs: Q[], go: (i: number) => void): HTMLElement {
  const card = h("div", "quiz quiz-summary");
  const done = progress.get(page.chapterId).quiz ?? {};
  const total = qs.filter((q) => done[q.id]?.correct).length;
  card.innerHTML = `<div class="sum-big"><b class="tabular">${total}</b><span>/ ${qs.length}</span></div><ul class="goal-list"></ul>`;
  const ul = card.querySelector("ul")!;
  page.objectives.forEach((text, gi) => {
    const li = h("li");
    const head = h("div", "goal");
    head.innerHTML = `<span class="gn">${gi + 1}</span><span></span>`;
    head.children[1].textContent = text;
    const links = h("div", "links");
    qs.forEach((q, qi) => {
      if (q.objective !== gi + 1) return;
      const b = h("button", `qdot ${done[q.id]?.correct ? "ok" : ""}`) as HTMLButtonElement;
      b.type = "button";
      b.textContent = String(qi + 1);
      b.setAttribute("aria-label", `문항 ${qi + 1}${done[q.id]?.correct ? " (맞힘)" : ""}`);
      b.addEventListener("click", () => go(qi));
      links.append(b);
    });
    head.append(links);
    li.append(head);
    ul.append(li);
  });
  return card;
}
