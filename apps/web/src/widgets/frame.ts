// 위젯 공통 틀(design-system.md §8): 머리(제목·종류) → 무대(canvas) → 조작 패널(슬라이더·프리셋·읽음값).
// 무대 배경은 테마와 무관하게 Midnight이고, 그 위의 선은 Glacier, 지금 볼 곳만 Coral로 칠한다.
export const C = {
  bg: "#0b1020", ink: "#dce6f2", faint: "rgba(220,230,242,0.14)", mid: "rgba(220,230,242,0.45)",
  edge: "#ff6b5a", edgeSoft: "rgba(255,107,90,0.18)", good: "#3fc79a", blue: "#9fc6ff",
};
const KIND: Record<string, string> = { manipulate: "직접 조작", stepper: "단계별 보기", simulate: "시뮬레이션", compare: "비교" };

export interface Frame {
  root: HTMLElement; stage: HTMLElement; canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; panel: HTMLElement;
  width: number; height: number; onResize: (fn: () => void) => void;
}

export function frame(root: HTMLElement, title: string, kind: string, aspect = 0.62, label = title): Frame {
  root.innerHTML = "";
  const wrap = el("div", "wframe");
  const head = el("div", "wframe-head");
  head.innerHTML = `<b></b><span class="kind">위젯 · ${KIND[kind] ?? kind}</span>`;
  head.querySelector("b")!.textContent = title;
  const stage = el("div", "wframe-stage");
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", label);
  stage.append(canvas);
  const panel = el("div", "wframe-panel");
  wrap.append(head, stage, panel);
  root.append(wrap);
  const ctx = canvas.getContext("2d")!;
  const listeners: (() => void)[] = [];
  const f: Frame = { root, stage, canvas, ctx, panel, width: 0, height: 0, onResize: (fn) => listeners.push(fn) };
  const fit = () => {
    const w = stage.clientWidth || 600;
    const h = Math.round(Math.min(Math.max(w * aspect, 240), 520));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    f.width = w;
    f.height = h;
    listeners.forEach((fn) => fn());
  };
  new ResizeObserver(() => fit()).observe(stage);
  fit();
  return f;
}

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", html?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/** 이름·슬라이더·값이 한 줄인 파라미터 조절기 */
export function param(opts: { label: string; min: number; max: number; step: number; value: number; fmt?: (v: number) => string; onInput: (v: number) => void; aria?: string }) {
  const row = el("div", "param");
  const id = `p-${Math.random().toString(36).slice(2, 8)}`;
  const lab = el("label");
  lab.htmlFor = id;
  lab.textContent = opts.label;
  const r = el("input");
  r.type = "range"; r.id = id;
  r.min = String(opts.min); r.max = String(opts.max); r.step = String(opts.step); r.value = String(opts.value);
  if (opts.aria) r.setAttribute("aria-label", opts.aria);
  const out = el("output");
  const fmt = opts.fmt ?? ((v: number) => v.toFixed(2));
  out.textContent = fmt(opts.value);
  r.addEventListener("input", () => { out.textContent = fmt(Number(r.value)); opts.onInput(Number(r.value)); });
  r.addEventListener("pointerdown", () => row.classList.add("is-active"));
  r.addEventListener("pointerup", () => row.classList.remove("is-active"));
  r.addEventListener("blur", () => row.classList.remove("is-active"));
  row.append(lab, r, out);
  return { row, set: (v: number) => { r.value = String(v); out.textContent = fmt(v); }, input: r };
}

/** 하나만 눌리는 프리셋 버튼 묶음 */
export function presets<T extends string>(names: { key: T; label: string }[], onPick: (k: T) => void, initial?: T) {
  const row = el("div", "wframe-row");
  row.setAttribute("role", "group");
  const btns = names.map(({ key, label }) => {
    const b = el("button", "preset");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(key === initial));
    b.addEventListener("click", () => { select(key); onPick(key); });
    row.append(b);
    return { key, b };
  });
  const select = (k: T | null) => btns.forEach(({ key, b }) => b.setAttribute("aria-pressed", String(key === k)));
  return { row, select };
}

export function readouts() {
  const row = el("div", "readouts");
  row.setAttribute("aria-live", "polite");
  const items = new Map<string, HTMLElement>();
  return {
    row,
    set(key: string, text: string, tone: "" | "hot" | "ok" = "") {
      let r = items.get(key);
      if (!r) { r = el("span", "readout"); items.set(key, r); row.append(r); }
      r.textContent = text;
      r.className = `readout ${tone}`;
    },
  };
}

/** 결정적 난수(시드 고정): 같은 시드는 언제나 같은 위젯 상태를 만든다. */
export function rng(seed: number) {
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => { const x = Math.max(u(), 1e-12); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * u()); };
  return { u, normal };
}

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
