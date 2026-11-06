// 탭 바(ADR 0009): 긴 목록·여러 패널을 한 번에 하나만 보인다. 선택 표시는 미끄러지고, 패널은 방향을 갖고 바뀐다.
// 마크업: [data-tabs] 안에 .tabbar > button[data-tab=키] 와 [data-panel=키]. 화살표 키로도 옮긴다.
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function tabs(root: HTMLElement, onChange?: (key: string) => void) {
  root.dataset.tabsReady = "true";
  const bar = root.querySelector<HTMLElement>(":scope > .tabbar")!;
  const buttons = [...bar.querySelectorAll<HTMLButtonElement>("button[data-tab]")];
  const panels = [...root.querySelectorAll<HTMLElement>(":scope > [data-panel]")];
  const ind = document.createElement("span");
  ind.className = "tab-ind";
  bar.append(ind);
  bar.setAttribute("role", "tablist");
  let cur = -1;
  const place = () => {
    const b = buttons[cur];
    if (!b || !b.offsetWidth) return;
    ind.style.width = `${b.offsetWidth}px`;
    ind.style.transform = `translateX(${b.offsetLeft}px)`;
  };
  function select(i: number, focus = false) {
    if (i === cur) return;
    const dir = i > cur ? 1 : -1;
    const first = cur < 0;
    cur = i;
    buttons.forEach((b, k) => {
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(k === i));
      b.tabIndex = k === i ? 0 : -1;
    });
    panels.forEach((p) => {
      const on = p.dataset.panel === buttons[i].dataset.tab;
      p.hidden = !on;
      p.setAttribute("role", "tabpanel");
      if (on && !first && !reduced()) {
        p.classList.remove("enter-next", "enter-prev");
        void p.offsetWidth;
        p.classList.add(dir > 0 ? "enter-next" : "enter-prev");
      }
    });
    if (focus) buttons[i].focus();
    place();
    onChange?.(buttons[i].dataset.tab!);
  }
  buttons.forEach((b, i) => {
    b.type = "button";
    b.addEventListener("click", () => select(i));
    b.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") { select((cur + 1) % buttons.length, true); e.preventDefault(); }
      if (e.key === "ArrowLeft") { select((cur - 1 + buttons.length) % buttons.length, true); e.preventDefault(); }
    });
  });
  new ResizeObserver(place).observe(bar);
  const initial = Math.max(0, buttons.findIndex((b) => b.dataset.selected === "true"));
  select(initial);
  return { select: (key: string) => { const i = buttons.findIndex((b) => b.dataset.tab === key); if (i >= 0) select(i); } };
}

export function mountAllTabs(scope: ParentNode = document) {
  scope.querySelectorAll<HTMLElement>("[data-tabs]:not([data-tabs-ready])").forEach((el) => tabs(el));
}
