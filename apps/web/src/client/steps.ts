// 장을 단계로 넘겨 읽는다(ADR 0009, editorial/benchmarks.md §3.1).
// 한 번에 한 절만 보인다. 이동: 상단 내비의 ←·→, 단계 막대, 드롭다운 목록, 하단 dock의 이전·다음, ←→ 키.
// 주소의 #은 지금 단계의 절 제목 id다. 다른 단계 안의 요소를 가리키는 링크를 누르면 그 단계로 넘어간다.
export function mountSteps() {
  const root = document.querySelector<HTMLElement>("[data-steps]");
  if (!root) return;
  const steps = [...root.querySelectorAll<HTMLElement>(":scope > section.step")];
  if (!steps.length) return;
  const bar = document.querySelector<HTMLElement>("[data-stepbar]");
  const list = document.querySelector<HTMLElement>("[data-steplist]");
  const prevBtns = [...document.querySelectorAll<HTMLButtonElement>("[data-prev]")];
  const nextBtns = [...document.querySelectorAll<HTMLButtonElement>("[data-next]")];
  const dockNext = document.querySelector<HTMLButtonElement>(".dock [data-next]");
  const count = document.querySelector<HTMLElement>("[data-stepcount]");
  const title = document.querySelector<HTMLElement>("[data-step-title]");
  const toggle = document.querySelector<HTMLButtonElement>("[data-chapnav-toggle]");
  const panel = document.querySelector<HTMLElement>("[data-chapnav-panel]");
  const done = document.querySelector<HTMLElement>("[data-done]");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const titles = steps.map((s) => s.dataset.title || "시작");
  const ids = steps.map((s) => s.querySelector("h2")?.id ?? "");
  const nextChapter = dockNext?.dataset.nextChapter;
  const seen = new Set<number>();
  let cur = -1;

  document.body.classList.add("paged");
  const segs = steps.map((_, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "seg";
    b.setAttribute("aria-label", titles[i]);
    b.dataset.tip = titles[i];
    b.addEventListener("click", () => show(i));
    bar?.append(b);
    return b;
  });
  const items = steps.map((_, i) => {
    if (!list) return null;
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = ids[i] ? `#${ids[i]}` : "#";
    a.innerHTML = `<span class="k tabular">${i + 1}</span><span></span>`;
    (a.lastChild as HTMLElement).textContent = titles[i];
    a.addEventListener("click", (e) => { e.preventDefault(); show(i); setPanel(false); });
    li.append(a);
    list.append(li);
    return a;
  });

  function setPanel(open: boolean) {
    if (!panel || !toggle) return;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open && !reduced) { panel.classList.remove("drop-in"); void panel.offsetWidth; panel.classList.add("drop-in"); }
  }
  toggle?.addEventListener("click", () => setPanel(panel?.hidden ?? false));
  document.addEventListener("click", (e) => {
    if (panel && !panel.hidden && !(e.target as HTMLElement).closest("[data-chapnav]")) setPanel(false);
  });

  function show(i: number, opts: { focus?: Element | null; initial?: boolean } = {}) {
    i = Math.max(0, Math.min(steps.length - 1, i));
    if (i === cur && !opts.focus) return;
    const dir = i > cur ? 1 : -1;
    const prev = steps[cur];
    cur = i;
    seen.add(i);
    steps.forEach((s, k) => { if (k !== i) s.hidden = true; });
    const el = steps[i];
    el.hidden = false;
    if (!reduced && !opts.initial && prev) {
      el.classList.remove("enter-next", "enter-prev");
      void el.offsetWidth;
      el.classList.add(dir > 0 ? "enter-next" : "enter-prev");
    }
    segs.forEach((b, k) => {
      b.setAttribute("aria-current", String(k === i));
      b.classList.toggle("seen", seen.has(k));
    });
    items.forEach((a, k) => a?.setAttribute("aria-current", String(k === i)));
    const last = i === steps.length - 1;
    prevBtns.forEach((b) => (b.disabled = i === 0));
    nextBtns.forEach((b) => (b.disabled = last && !(b === dockNext && nextChapter)));
    if (dockNext) dockNext.textContent = last ? (nextChapter ? "다음 장" : "끝") : "다음";
    if (count) count.textContent = `${i + 1} / ${steps.length}`;
    if (title) title.textContent = titles[i];
    if (done) done.hidden = !last;
    history.replaceState(null, "", ids[i] && i > 0 ? `#${ids[i]}` : location.pathname);
    if (opts.focus) opts.focus.scrollIntoView({ block: "start" });
    else if (!opts.initial) {
      const nav = document.querySelector<HTMLElement>("[data-chapnav]");
      const top = (nav?.getBoundingClientRect().top ?? 0) + scrollY - 8;
      if (scrollY > top) scrollTo({ top, behavior: reduced ? "auto" : "smooth" });
    }
    window.dispatchEvent(new CustomEvent("step-change", { detail: { index: i } }));
  }

  prevBtns.forEach((b) => b.addEventListener("click", () => show(cur - 1)));
  nextBtns.forEach((b) => b.addEventListener("click", () => {
    if (cur === steps.length - 1 && b === dockNext && nextChapter) location.href = nextChapter;
    else show(cur + 1);
  }));
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    if (e.key === "Escape") setPanel(false);
    if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, .cm-editor, [contenteditable], dialog, canvas")) return;
    if (e.key === "ArrowRight") show(cur + 1);
    else if (e.key === "ArrowLeft") show(cur - 1);
  });

  const stepOf = (el: Element | null) => (el ? steps.findIndex((s) => s.contains(el)) : -1);
  const fromHash = (hash: string, initial = false) => {
    const target = hash ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
    const k = stepOf(target);
    if (k < 0) return false;
    show(k, { focus: target?.tagName === "H2" ? null : target, initial });
    return true;
  };
  document.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).closest<HTMLAnchorElement>('a[href^="#"]');
    if (!a || a.closest("[data-steplist]")) return;
    if (fromHash(a.hash)) e.preventDefault();
  });
  window.addEventListener("hashchange", () => fromHash(location.hash));
  if (!fromHash(location.hash, true)) show(0, { initial: true });

  // 다른 모듈이 특정 요소가 있는 단계로 넘길 때(예: 퀴즈 오답 → 관련 절)
  window.addEventListener("show-step-of", (e) => { const k = stepOf((e as CustomEvent).detail); if (k >= 0) show(k); });
}
