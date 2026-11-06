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
  dialog.append(close, renderQuestion(chapterId, q, { onContinue: () => dialog!.close(), continueLabel: "계속 읽기" }));
  dialog.showModal();
}

/** 개념 퀴즈 단계: 한 번에 한 문항. 위쪽 막대가 진행과 정답을 보여 준다. 마지막은 학습 목표별 요약. */
function mountCarousel(root: HTMLElement, page: PageData, qs: Q[]) {
  const bar = h("div", "qbar");
  const stage = h("div", "qstage");
  root.append(bar, stage);
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
    const card = i < qs.length
      ? renderQuestion(page.chapterId, qs[i], { onContinue: () => go(i + 1), continueLabel: i === qs.length - 1 ? "결과 보기" : "다음 문항", related: true })
      : summary(page, qs, go);
    if (!reduced) card.classList.add(dir > 0 ? "enter-next" : "enter-prev");
    stage.append(card);
    paint();
  }
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

interface QOpts { onContinue?: () => void; continueLabel?: string; related?: boolean }

/** 문항 카드. 하단 주 버튼 하나가 '확인' → (정답) '계속' / (오답) '다시'로 바뀐다 (benchmarks.md §3.3). */
function renderQuestion(chapterId: string, q: Q, opts: QOpts = {}): HTMLElement {
  const box = h("div", "quiz");
  box.id = `quiz-${q.id}`;
  box.dataset.quizId = q.id;
  if (q.type === "multiple") box.append(h("div", "quiz-head", `<span class="tag">모두 고르기</span>`));
  box.append(h("div", "quiz-prompt", q.prompt));
  if (progress.get(chapterId).quiz?.[q.id]?.correct) box.classList.add("solved");

  const fb = h("div", "quiz-fb");
  fb.hidden = true;
  fb.setAttribute("aria-live", "polite");
  const primary = h("button", "btn btn-primary quiz-primary", "확인") as HTMLButtonElement;
  primary.type = "button";
  primary.disabled = true;
  let phase: "answer" | "right" | "wrong" = "answer";
  let read: () => { ok: boolean; extra: string } | null = () => null;
  let clearMarks = () => {};

  const setPhase = (ph: typeof phase) => {
    phase = ph;
    primary.classList.toggle("is-right", ph === "right");
    primary.classList.toggle("is-wrong", ph === "wrong");
    if (ph === "answer") { primary.textContent = "확인"; fb.hidden = true; }
    if (ph === "right") { primary.textContent = opts.continueLabel ?? "계속"; primary.disabled = !opts.onContinue; }
    if (ph === "wrong") { primary.textContent = "다시"; primary.disabled = false; }
  };
  const answered = () => { if (phase !== "answer") { clearMarks(); setPhase("answer"); } primary.disabled = false; };

  primary.addEventListener("click", () => {
    if (phase === "right") { opts.onContinue?.(); return; }
    if (phase === "wrong") { clearMarks(); setPhase("answer"); primary.disabled = true; return; }
    const r = read();
    if (!r) return;
    fb.hidden = false;
    fb.className = `quiz-fb ${r.ok ? "good" : ""}`;
    fb.innerHTML = r.ok ? `<span class="lbl">정답</span>${q.explanation}` : `<span class="lbl">다시 생각해 보자</span>${r.extra}`;
    if (!r.ok && opts.related) {
      const anchor = document.querySelector(`.quiz-open[data-quiz-open="${q.id}"]`);
      if (anchor) {
        const a = h("button", "quiz-related", "이 내용을 다룬 절로 →") as HTMLButtonElement;
        a.type = "button";
        a.addEventListener("click", () => {
          window.dispatchEvent(new CustomEvent("show-step-of", { detail: anchor }));
          setTimeout(() => anchor.scrollIntoView({ block: "center" }), 60);
        });
        fb.append(a);
      }
    }
    record(chapterId, q.id, r.ok);
    if (r.ok) box.classList.add("solved");
    setPhase(r.ok ? "right" : "wrong");
  });

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
        if (q.type === "single") buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        else b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
        if (!buttons.some((x) => x.getAttribute("aria-pressed") === "true")) { primary.disabled = true; return; }
        answered();
      });
      buttons.push(b);
      li.append(b);
      ul.append(li);
    });
    box.append(ul);
    clearMarks = () => buttons.forEach((x) => x.classList.remove("is-wrong", "is-right"));
    read = () => {
      const picked = buttons.map((b) => b.getAttribute("aria-pressed") === "true");
      if (!picked.some(Boolean)) return null;
      const wrong: string[] = [];
      let missed = 0;
      q.choices.forEach((c, i) => {
        if (picked[i] && !c.correct) { void buttons[i].offsetWidth; buttons[i].classList.add("is-wrong"); wrong.push(q.type === "multiple" ? `<b>${KEYS[i]}</b> ${c.feedback || "옳지 않은 진술이다."}` : (c.feedback || "이 선택지가 왜 틀렸는지 문항의 조건을 다시 읽어 보자.")); }
        if (picked[i] && c.correct) buttons[i].classList.add("is-right");
        if (!picked[i] && c.correct) missed++;
      });
      if (missed && q.type === "multiple") wrong.push(`옳은 진술 ${missed}개를 더 골라야 한다.`);
      return { ok: !wrong.length && !missed, extra: wrong.join("<br />") };
    };
  } else {
    const wrap = h("div", "quiz-numeric");
    const s = q.slider;
    const unit = s?.unit ?? "";
    const num = h("input") as HTMLInputElement;
    num.type = "number";
    num.setAttribute("aria-label", "답");
    if (s) {
      const row = h("div", "quiz-slider");
      const range = h("input") as HTMLInputElement;
      range.type = "range";
      range.min = String(s.min); range.max = String(s.max); range.step = String(s.step);
      range.value = String(s.min);
      range.setAttribute("aria-label", "슬라이더로 답 고르기");
      const out = h("output");
      const sync = (v: string) => { out.textContent = `${v}${unit}`; };
      range.addEventListener("input", () => { num.value = range.value; sync(range.value); answered(); });
      num.addEventListener("input", () => { if (num.value !== "") { range.value = num.value; sync(num.value); } });
      sync(range.value);
      num.value = range.value;
      num.min = range.min; num.max = range.max; num.step = range.step;
      row.append(range, out);
      wrap.append(row);
    }
    num.addEventListener("input", () => (num.value === "" ? (primary.disabled = true) : answered()));
    num.addEventListener("keydown", (e) => { if (e.key === "Enter" && !primary.disabled) primary.click(); });
    wrap.append(num);
    box.append(wrap);
    read = () => {
      const v = Number(num.value);
      if (num.value === "" || Number.isNaN(v)) return null;
      const ok = Math.abs(v - (q.answer ?? NaN)) <= (q.tolerance ?? 0) + 1e-12;
      return { ok, extra: v < (q.answer ?? 0) ? "그보다 크다." : "그보다 작다." };
    };
  }
  box.append(fb, h("div", "quiz-actions"));
  box.lastElementChild!.append(primary);
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
