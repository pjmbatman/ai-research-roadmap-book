// 레슨 화면의 단계 이동 (Brilliant·Duolingo, design-system.md §3, ADR 0009).
// 한 번에 한 절(단계)만 보인다. 위 레슨 바: 진행 막대 · k/N · 단계 목록. 아래 막대: 이전 · 계속(마지막에는 다음 장).
// ←→ 키로도 넘긴다. 주소의 #은 지금 단계의 절 제목 id다.
export function mountSteps() {
  const root = document.querySelector<HTMLElement>("[data-steps]");
  if (!root) return;
  const steps = [...root.querySelectorAll<HTMLElement>(":scope > section.step")];
  if (!steps.length) return;
  const article = root.closest<HTMLElement>(".lesson");
  const bar = document.querySelector<HTMLElement>("[data-lesson-progress] > i");
  const count = document.querySelector<HTMLElement>("[data-stepcount]");
  const list = document.querySelector<HTMLElement>("[data-steplist]");
  const menu = document.querySelector<HTMLElement>("[data-lesson-menu]");
  const toggle = document.querySelector<HTMLButtonElement>("[data-lesson-menu-toggle]");
  const prevBtn = document.querySelector<HTMLButtonElement>("[data-prev]");
  const nextBtn = document.querySelector<HTMLButtonElement>("[data-next]");
  const done = document.querySelector<HTMLElement>("[data-done]");
  const nextChapter = nextBtn?.dataset.nextChapter;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const titles = steps.map((s) => s.dataset.title || "시작");
  const ids = steps.map((s) => s.querySelector("h2")?.id ?? "");
  let cur = -1;

  document.body.classList.add("paged");
  const items = steps.map((_, i) => {
    if (!list) return null;
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = ids[i] ? `#${ids[i]}` : "#";
    a.innerHTML = `<span class="k tabular">${i + 1}</span><span></span>`;
    (a.lastChild as HTMLElement).textContent = titles[i];
    a.addEventListener("click", (e) => { e.preventDefault(); show(i); setMenu(false); });
    li.append(a);
    list.append(li);
    return a;
  });

  function setMenu(open: boolean) {
    if (!menu || !toggle) return;
    menu.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
  }
  toggle?.addEventListener("click", () => setMenu(menu?.hidden ?? false));
  document.addEventListener("click", (e) => {
    if (menu && !menu.hidden && !(e.target as HTMLElement).closest("[data-lesson-menu], [data-lesson-menu-toggle]")) setMenu(false);
  });

  function show(i: number, opts: { focus?: Element | null; initial?: boolean } = {}) {
    i = Math.max(0, Math.min(steps.length - 1, i));
    if (i === cur && !opts.focus) return;
    const dir = i > cur ? 1 : -1;
    const had = cur >= 0;
    cur = i;
    steps.forEach((s, k) => { s.hidden = k !== i; });
    const el = steps[i];
    if (!reduced && had && !opts.initial) {
      el.classList.remove("enter-next", "enter-prev");
      void el.offsetWidth;
      el.classList.add(dir > 0 ? "enter-next" : "enter-prev");
    }
    if (article) article.dataset.step = String(i);
    items.forEach((a, k) => a?.setAttribute("aria-current", String(k === i)));
    if (bar) bar.style.width = `${((i + 1) / steps.length) * 100}%`;
    bar?.parentElement?.setAttribute("aria-valuenow", String(Math.round(((i + 1) / steps.length) * 100)));
    if (count) count.textContent = `${i + 1} / ${steps.length}`;
    const last = i === steps.length - 1;
    if (prevBtn) prevBtn.disabled = i === 0;
    if (nextBtn) {
      nextBtn.textContent = last ? (nextChapter ? "다음 장" : "끝") : "계속";
      nextBtn.disabled = last && !nextChapter;
    }
    if (done) done.hidden = !last;
    history.replaceState(null, "", ids[i] && i > 0 ? `#${ids[i]}` : location.pathname);
    if (opts.focus) opts.focus.scrollIntoView({ block: "center" });
    else if (!opts.initial) scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    window.dispatchEvent(new CustomEvent("step-change", { detail: { index: i } }));
  }

  prevBtn?.addEventListener("click", () => show(cur - 1));
  nextBtn?.addEventListener("click", () => {
    if (cur === steps.length - 1 && nextChapter) location.href = nextChapter;
    else show(cur + 1);
  });
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    if (e.key === "Escape") setMenu(false);
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

  // 다른 모듈이 특정 요소가 있는 단계로 넘길 때(퀴즈 오답 → 관련 절)
  window.addEventListener("show-step-of", (e) => { const k = stepOf((e as CustomEvent).detail); if (k >= 0) show(k); });
}
