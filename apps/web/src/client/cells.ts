// 본문 코드 셀. exec="browser"는 편집·실행할 수 있고, 같은 session의 셀은 변수를 공유한다.
// 앞 셀을 실행하지 않고 뒤 셀을 실행하면 같은 session의 앞 셀을 먼저 차례로 실행한다.
// exec="python"·"gpu"는 로컬 환경이 필요하므로 읽기 전용으로 두고 복사 버튼만 단다.
import { createEditor } from "./editor";
import { session } from "./runtime";

interface Cell { el: HTMLElement; session: string; get: () => string; original: string; ran: boolean; run: (opts?: { quiet?: boolean }) => Promise<boolean> }
const cells: Cell[] = [];
let queue: Promise<unknown> = Promise.resolve();

export function mountCells() {
  document.querySelectorAll<HTMLElement>(".cell").forEach((el) => {
    const exec = el.dataset.exec || "browser";
    const src = el.querySelector<HTMLElement>("pre.cell-src")!;
    const out = el.querySelector<HTMLElement>("pre.cell-out")!;
    const bar = el.querySelector<HTMLElement>(".cell-bar")!;
    const original = src.textContent || "";
    if (exec !== "browser") {
      const tag = Object.assign(document.createElement("span"), { className: "cell-tag", textContent: exec === "gpu" ? "GPU" : "로컬 Python" });
      bar.append(tag, spacer(), copyButton(() => original));
      return;
    }

    const status = Object.assign(document.createElement("span"), { className: "cell-status" });
    status.setAttribute("aria-live", "polite");
    const runBtn = Object.assign(document.createElement("button"), { className: "run", type: "button", textContent: "▶ 실행" }) as HTMLButtonElement;
    runBtn.title = "Shift+Enter";
    const resetBtn = iconButton(ICON.reset, "원래 코드로 되돌리기");
    bar.append(status, spacer(), resetBtn, copyButton(() => cell.get()), runBtn);

    const host = document.createElement("div");
    src.replaceWith(host);
    const ed = createEditor(host, original.replace(/\n$/, ""), { onRun: () => runBtn.click(), label: "코드 셀 편집기" });

    const cell: Cell = {
      el, session: el.dataset.session || "default", get: ed.get, original, ran: false,
      run: async ({ quiet = false } = {}) => {
        runBtn.disabled = true;
        runBtn.textContent = "실행 중"; status.textContent = "";
        const t0 = performance.now();
        const live: string[] = [];
        out.hidden = false;
        if (!quiet) { out.dataset.recorded = "false"; out.dataset.live = "true"; out.classList.remove("error"); out.textContent = ""; }
        const r = await session(cell.session).run(ed.get(), { onStdout: (s) => { if (!quiet) { live.push(s); out.textContent = live.join(""); } } });
        runBtn.disabled = false;
        cell.ran = r.ok;
        runBtn.textContent = "▶ 실행";
        if (quiet && r.ok) return true;
        out.dataset.recorded = "false";
        out.dataset.live = "true";
        out.textContent = (r.stdout || "") + (r.ok ? "" : (r.stdout ? "\n" : "") + cleanError(r.error || ""));
        if (!out.textContent) out.textContent = "(출력 없음)";
        out.classList.toggle("error", !r.ok);
        status.textContent = r.ok ? "✓" : "";
        status.dataset.ok = String(r.ok);
        void t0;
        return r.ok;
      },
    };
    cells.push(cell);

    runBtn.addEventListener("click", () => {
      queue = queue.then(async () => {
        const mine = cells.filter((c) => c.session === cell.session);
        for (const c of mine.slice(0, mine.indexOf(cell))) {
          if (!c.ran && !(await c.run({ quiet: true }))) { status.textContent = "앞 셀 오류"; return; }
        }
        await cell.run();
      });
    });
    resetBtn.addEventListener("click", () => {
      ed.set(original.replace(/\n$/, ""));
      out.classList.remove("error");
      status.textContent = "";
    });
  });
}

const ICON = {
  reset: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>',
};
const spacer = () => Object.assign(document.createElement("span"), { className: "spacer" });
export function iconButton(svg: string, label: string) {
  const b = Object.assign(document.createElement("button"), { type: "button", className: "icon" }) as HTMLButtonElement;
  b.innerHTML = svg;
  b.setAttribute("aria-label", label);
  b.title = label;
  return b;
}
function copyButton(get: () => string) {
  const b = iconButton(ICON.copy, "코드 복사");
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(get()); b.innerHTML = ICON.check; } catch {}
    setTimeout(() => (b.innerHTML = ICON.copy), 1200);
  });
  return b;
}

/** Pyodide 오류에서 내부 프레임을 걷어내고 사용자 코드에 해당하는 부분만 남긴다. */
export function cleanError(err: string): string {
  const lines = err.split("\n");
  const i = lines.findIndex((l) => l.includes('File "<exec>"'));
  return (i >= 0 ? ["Traceback (most recent call last):", ...lines.slice(i)] : lines).join("\n").trim();
}
