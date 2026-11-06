// 위젯 틀 (design-system.md §4): Card 안에 머리(제목·초기화) → 그림(canvas) → 조작(ToggleGroup·Slider·Switch·읽음값).
// 그림의 색은 디자인 토큰에서 읽어 라이트·다크 테마를 따른다.
// 색은 디자인 토큰(design-system.md §4)에서 읽는다: 바탕 = --card, 축·격자 = --border·--muted-foreground,
// 주된 데이터 = --chart-3, 보조 데이터 = --chart-1. 테마가 바뀌면 다시 읽고 다시 그린다.
export const C = { bg: "", ink: "", mid: "", faint: "", edge: "", ink2: "", good: "", edgeSoft: "", grid2: "", fill: "" };
const probe = typeof document !== "undefined" ? document.createElement("canvas").getContext("2d", { willReadFrequently: true }) : null;
/** CSS 색(oklch 포함)을 [r, g, b]로 바꾼다. */
export function rgbOf(css: string): [number, number, number] {
  if (!probe) return [0, 0, 0];
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#000";
  probe.fillStyle = css;
  probe.fillRect(0, 0, 1, 1);
  const d = probe.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}
export const alpha = (css: string, a: number) => { const [r, g, b] = rgbOf(css); return `rgba(${r},${g},${b},${a})`; };
export function refreshColors() {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  Object.assign(C, {
    bg: v("--card"), ink: v("--foreground"), mid: v("--muted-foreground"), faint: v("--border"),
    edge: v("--chart-3"), ink2: v("--chart-1"), good: v("--success"),
    edgeSoft: alpha(v("--chart-1"), 0.35), grid2: alpha(v("--muted-foreground"), 0.35), fill: alpha(v("--chart-1"), 0.25),
  });
}
if (typeof window !== "undefined") {
  refreshColors();
  const again = () => { refreshColors(); window.dispatchEvent(new CustomEvent("widget-recolor")); };
  window.addEventListener("theme-change", again);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", again);
}

export interface Frame {
  root: HTMLElement; stage: HTMLElement; canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; panel: HTMLElement;
  width: number; height: number; onResize: (fn: () => void) => void;
}

export function frame(root: HTMLElement, title: string, kind: string, aspect = 0.62, label = title): Frame {
  root.innerHTML = "";
  const wrap = el("div", "wframe");
  const head = el("div", "wframe-head");
  head.innerHTML = `<b></b>`;
  void kind;
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
  window.addEventListener("widget-recolor", () => listeners.forEach((fn) => fn()));
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
