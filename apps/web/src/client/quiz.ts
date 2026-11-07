// 개념 퀴즈 문항 카드(editorial/conventions.md §9). 퀘스트의 퀴즈 노드가 시트 안에 한 문항씩 띄운다(ADR 0012).
// 정답·해설은 공개 데이터다(로그인 없는 오픈소스, AGENTS.md). 채점은 학습을 위한 즉시 피드백이다.
import { progress } from "./store";

export interface Q {
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

function record(chapterId: string, id: string, correct: boolean) {
  progress.update(chapterId, (p) => {
    const prev = (p.quiz ??= {})[id];
    p.quiz[id] = { correct: correct || !!prev?.correct, tries: (prev?.tries ?? 0) + 1 };
  });
}

interface QOpts { onContinue?: () => void; continueLabel?: string }

/** 문항 카드. 하단 주 버튼 하나가 '확인' → (정답) '계속' / (오답) '다시'로 바뀐다 (benchmarks.md §3.3). */
export function renderQuestion(chapterId: string, q: Q, opts: QOpts = {}): HTMLElement {
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
