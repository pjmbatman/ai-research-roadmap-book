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
    bar.append(Object.assign(document.createElement("span"), { className: "spacer" }));

    if (exec !== "browser") {
      const note = document.createElement("span");
      note.className = "cell-status";
      note.textContent = exec === "gpu" ? "GPU가 있는 로컬 환경에서 실행" : "로컬 Python에서 실행(PyTorch 등)";
      bar.append(note, copyButton(() => original));
      return;
    }

    const status = Object.assign(document.createElement("span"), { className: "cell-status" });
    status.setAttribute("aria-live", "polite");
    const runBtn = Object.assign(document.createElement("button"), { className: "run", type: "button", textContent: "▶ 실행" }) as HTMLButtonElement;
    runBtn.title = "실행 (Shift+Enter)";
    const resetBtn = Object.assign(document.createElement("button"), { type: "button", textContent: "되돌리기" }) as HTMLButtonElement;
    resetBtn.title = "원래 코드로 되돌리기";
    bar.append(status, resetBtn, copyButton(() => cell.get()), runBtn);

    const host = document.createElement("div");
    src.replaceWith(host);
    const ed = createEditor(host, original.replace(/\n$/, ""), { onRun: () => runBtn.click(), label: "코드 셀 편집기" });

    const cell: Cell = {
      el, session: el.dataset.session || "default", get: ed.get, original, ran: false,
      run: async ({ quiet = false } = {}) => {
        runBtn.disabled = true;
        status.textContent = quiet ? "앞 셀 실행 중…" : "실행 중…";
        const t0 = performance.now();
        const live: string[] = [];
        out.hidden = false;
        if (!quiet) { out.dataset.recorded = "false"; out.dataset.live = "true"; out.classList.remove("error"); out.textContent = ""; }
        const r = await session(cell.session).run(ed.get(), { onStdout: (s) => { if (!quiet) { live.push(s); out.textContent = live.join(""); } } });
        runBtn.disabled = false;
        cell.ran = r.ok;
        if (quiet && r.ok) { status.textContent = "실행됨"; return true; }
        out.dataset.recorded = "false";
        out.dataset.live = "true";
        out.textContent = (r.stdout || "") + (r.ok ? "" : (r.stdout ? "\n" : "") + cleanError(r.error || ""));
        if (!out.textContent) out.textContent = "(출력 없음)";
        out.classList.toggle("error", !r.ok);
        status.textContent = r.ok ? `완료 · ${((performance.now() - t0) / 1000).toFixed(1)}초` : "오류";
        return r.ok;
      },
    };
    cells.push(cell);

    runBtn.addEventListener("click", () => {
      queue = queue.then(async () => {
        const mine = cells.filter((c) => c.session === cell.session);
        for (const c of mine.slice(0, mine.indexOf(cell))) {
          if (!c.ran && !(await c.run({ quiet: true }))) { status.textContent = "앞 셀에서 오류"; return; }
        }
        await cell.run();
      });
    });
    resetBtn.addEventListener("click", () => {
      ed.set(original.replace(/\n$/, ""));
      out.classList.remove("error");
      status.textContent = "원래 코드";
    });
  });
}

function copyButton(get: () => string) {
  const b = Object.assign(document.createElement("button"), { type: "button", textContent: "복사" }) as HTMLButtonElement;
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(get()); b.textContent = "복사됨"; } catch { b.textContent = "복사 실패"; }
    setTimeout(() => (b.textContent = "복사"), 1400);
  });
  return b;
}

/** Pyodide 오류에서 내부 프레임을 걷어내고 사용자 코드에 해당하는 부분만 남긴다. */
export function cleanError(err: string): string {
  const lines = err.split("\n");
  const i = lines.findIndex((l) => l.includes('File "<exec>"'));
  return (i >= 0 ? ["Traceback (most recent call last):", ...lines.slice(i)] : lines).join("\n").trim();
}
