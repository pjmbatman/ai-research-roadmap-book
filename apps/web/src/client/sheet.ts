// 서랍(Sheet, shadcn/ui): 오른쪽에서 나오는 패널. 바깥·Esc·닫기 버튼으로 닫고, 닫으면 연 버튼으로 초점을 돌려준다.
let current: { close: () => void } | null = null;

export function openSheet(build: (panel: HTMLElement, close: () => void) => void, label: string) {
  current?.close();
  const opener = document.activeElement as HTMLElement | null;
  const backdrop = document.createElement("div");
  backdrop.className = "sheet-backdrop";
  const panel = document.createElement("aside");
  panel.className = "sheet";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "true");
  panel.setAttribute("aria-label", label);
  const x = document.createElement("button");
  x.type = "button";
  x.className = "icon-btn sheet-close";
  x.setAttribute("aria-label", "닫기");
  x.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>';
  const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
  const close = () => {
    backdrop.remove(); panel.remove();
    document.removeEventListener("keydown", onKey);
    current = null;
    opener?.focus?.();
  };
  x.addEventListener("click", close);
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  panel.append(x);
  build(panel, close);
  document.body.append(backdrop, panel);
  (panel.querySelector<HTMLElement>("a.btn-primary, button:not(.sheet-close)") ?? x).focus();
  current = { close };
  return { close };
}

export const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
