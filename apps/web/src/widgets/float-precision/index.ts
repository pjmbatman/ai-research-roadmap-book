// 부동소수점 비트 구조 (data/widgets.toml: float-precision).
// 수 x를 float32 · float16 · bfloat16으로 반올림(최근접, 동률이면 짝수)해 부호·지수·가수 비트와 저장값, 반올림 오차, ulp를 보인다.
// 세 형식의 비트 칸은 같은 폭으로 그려 bfloat16이 float32의 앞 16비트와 같은 배치임이 보이게 한다.
// 조작: 프리셋·크기 슬라이더·부호 스위치로 x를 고르거나, 그림의 비트 칸을 눌러 그 형식의 비트를 직접 뒤집는다.
import { frame, param, presets, readouts, el, C } from "../frame";
import { pageState } from "../../client/store";

export interface Fmt { key: string; name: string; e: number; m: number }
export const FMTS: Fmt[] = [
  { key: "f32", name: "float32", e: 8, m: 23 },
  { key: "f16", name: "float16", e: 5, m: 10 },
  { key: "bf16", name: "bfloat16", e: 8, m: 7 },
];
const bias = (F: Fmt) => 2 ** (F.e - 1) - 1;
const emin = (F: Fmt) => 1 - bias(F);
const maxFinite = (F: Fmt) => (2 - 2 ** -F.m) * 2 ** bias(F);
/** ⌊log₂ a⌋를 정확히(a > 0) */
function ilog2(a: number) {
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  return e;
}
const rne = (v: number) => { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1; };

/** double x를 형식 F의 가장 가까운 값으로(동률이면 가수가 짝수인 쪽, 넘치면 ±inf) */
export function roundTo(x: number, F: Fmt): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  const e = Math.max(ilog2(a), emin(F));
  const q = 2 ** (e - F.m); // 이 구간의 ulp
  const r = rne(a / q) * q;
  if (r > maxFinite(F)) return x < 0 ? -Infinity : Infinity;
  return x < 0 ? -r : r; // r = 0이면 −0을 지킨다
}
/** 형식 F의 값 r(roundTo의 결과)을 비트로: [부호, 지수 e개, 가수 m개] */
export function encode(r: number, F: Fmt): number[] {
  const sign = r < 0 || Object.is(r, -0) ? 1 : 0;
  const Emax = 2 ** F.e - 1;
  let E = 0, M = 0;
  if (Number.isNaN(r)) { E = Emax; M = 2 ** (F.m - 1); }
  else if (!Number.isFinite(r)) { E = Emax; }
  else if (r !== 0) {
    const a = Math.abs(r);
    if (a < 2 ** emin(F)) M = a / 2 ** (emin(F) - F.m);
    else { const e = ilog2(a); E = e + bias(F); M = (a / 2 ** e - 1) * 2 ** F.m; }
  }
  const bits = [sign];
  for (let i = F.e - 1; i >= 0; i--) bits.push(Math.floor(E / 2 ** i) % 2);
  for (let i = F.m - 1; i >= 0; i--) bits.push(Math.floor(M / 2 ** i) % 2);
  return bits;
}
export function decode(bits: number[], F: Fmt): number {
  const s = bits[0] ? -1 : 1;
  let E = 0, M = 0;
  for (let i = 1; i <= F.e; i++) E = E * 2 + bits[i];
  for (let i = F.e + 1; i < bits.length; i++) M = M * 2 + bits[i];
  if (E === 2 ** F.e - 1) return M ? NaN : s * Infinity;
  if (E === 0) return s * M * 2 ** (emin(F) - F.m);
  return s * (1 + M / 2 ** F.m) * 2 ** (E - bias(F));
}
/** r이 있는 구간의 ulp(이웃한 두 수의 간격) */
export function ulp(r: number, F: Fmt) {
  if (!Number.isFinite(r)) return NaN;
  const a = Math.abs(r);
  return 2 ** (Math.max(a > 0 ? ilog2(a) : emin(F), emin(F)) - F.m);
}

const SUP: Record<string, string> = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = (n: number) => String(n).split("").map((c) => SUP[c] ?? c).join("");
const sci = (v: number, d = 2) => (v === 0 ? "0" : v.toExponential(d).replace("e+", "e").replace(/-/g, "−"));
const show = (v: number) => (Number.isNaN(v) ? "NaN" : !Number.isFinite(v) ? (v < 0 ? "−inf" : "inf") : Object.is(v, -0) ? "−0" : String(Number(v.toPrecision(17))).replace("-", "−"));

const PRESETS: { key: string; label: string; x: number }[] = [
  { key: "0.1", label: "0.1", x: 0.1 },
  { key: "third", label: "1/3", x: 1 / 3 },
  { key: "2049", label: "2049", x: 2049 },
  { key: "2p24", label: "2²⁴+1", x: 2 ** 24 + 1 },
  { key: "65504", label: "65504", x: 65504 },
  { key: "1e5", label: "1e5", x: 1e5 },
  { key: "1e-5", label: "1e−5", x: 1e-5 },
  { key: "1e-8", label: "1e−8", x: 1e-8 },
];

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "부동소수점 비트 구조", "manipulate", root.clientWidth < 480 ? 1.05 : 0.6, "float32, float16, bfloat16의 부호·지수·가수 비트와 저장값, 반올림 오차");
  const start = PRESETS.find((p) => p.key === attrs.preset) ?? PRESETS[0];
  let x = start.x;
  let hover: [number, number] | null = null;

  function geom() {
    const pad = 12;
    const cw = Math.min(17, (f.width - 2 * pad) / 32);
    const ch = Math.min(22, Math.max(14, cw * 1.5));
    const fs = Math.max(10, Math.min(12.5, f.width / 42));
    const rowH = (f.height - 8) / 3;
    const rows = FMTS.map((F, i) => ({ F, y: 4 + i * rowH }));
    return { pad, cw, ch, fs, rowH, rows };
  }
  const bitBox = (g: ReturnType<typeof geom>, i: number, j: number) => {
    const { pad, cw, ch, fs, rows } = g;
    return { x: pad + j * cw, y: rows[i].y + fs + 10, w: cw, h: ch };
  };

  function draw() {
    const { ctx, width: W } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, f.height);
    const g = geom();
    const { pad, cw, ch, fs, rows } = g;
    rows.forEach(({ F, y }, i) => {
      const r = roundTo(x, F);
      const bits = encode(r, F);
      // 머리: 형식 이름과 저장값
      ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.ink;
      ctx.fillText(F.name, pad, y + fs + 2);
      const nameW = ctx.measureText(F.name + " ").width;
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.mid;
      ctx.fillText(`1·${F.e}·${F.m}`, pad + nameW, y + fs + 2);
      const val = `= ${show(r)}`;
      ctx.fillStyle = Number.isFinite(r) && (r !== 0 || x === 0) ? C.ink : C.edge;
      ctx.font = `600 ${fs}px JetBrains Mono, monospace`;
      ctx.textAlign = "right";
      ctx.fillText(val, W - pad, y + fs + 2);
      ctx.textAlign = "start";
      // 비트 칸: 1은 채우고 0은 테두리만
      bits.forEach((b, j) => {
        const bx = bitBox(g, i, j);
        const col = j === 0 ? C.ink : j <= F.e ? C.edge : C.ink2;
        const hv = hover && hover[0] === i && hover[1] === j;
        ctx.fillStyle = b ? col : C.bg;
        ctx.fillRect(bx.x + 1, bx.y, bx.w - 2, bx.h);
        ctx.strokeStyle = hv ? C.ink : col;
        ctx.lineWidth = hv ? 2 : 1;
        ctx.strokeRect(bx.x + 1.5, bx.y + 0.5, bx.w - 3, bx.h - 1);
        if (cw >= 13) {
          ctx.fillStyle = b ? C.bg : C.mid;
          ctx.font = `600 ${Math.min(11, cw * 0.7)}px JetBrains Mono, monospace`;
          ctx.textAlign = "center";
          ctx.fillText(String(b), bx.x + bx.w / 2, bx.y + bx.h / 2 + 4);
          ctx.textAlign = "start";
        }
      });
      // 해석: 2^e × 1.f, 그리고 오차·ulp
      const by = y + fs + 10 + ch;
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.mid;
      let dec = "";
      if (Number.isNaN(r)) dec = "NaN";
      else if (!Number.isFinite(r)) dec = "지수 비트가 모두 1 → inf";
      else if (r === 0) dec = "지수·가수 비트가 모두 0 → 0";
      else {
        const a = Math.abs(r), sg = r < 0 ? "−" : "+";
        if (a < 2 ** emin(F)) dec = `${sg}2${sup(emin(F))} × ${(a / 2 ** emin(F)).toPrecision(8)} (서브노멀)`;
        else { const e = ilog2(a); dec = `${sg}2${sup(e)} × ${(a / 2 ** e).toPrecision(F.m > 10 ? 10 : 6)}`; }
      }
      ctx.fillText(dec, pad, by + fs + 4);
      const err = r - x, u = ulp(r, F);
      const errTxt = Number.isFinite(r) && Number.isFinite(x)
        ? `오차 ${err >= 0 ? "+" : ""}${sci(err)} · ulp ${sci(u)}`
        : "오차 —";
      ctx.fillStyle = Number.isFinite(r) && Math.abs(err) <= u / 2 ? C.mid : C.edge;
      ctx.fillText(errTxt, pad, by + 2 * fs + 9);
    });
  }

  // 조작
  const pre = presets(PRESETS.map((p) => ({ key: p.key, label: p.label })), (k) => { x = PRESETS.find((p) => p.key === k)!.x; sync(); update(); }, start.key);
  const pm = param({
    label: "log₁₀|x|", min: -9, max: 8, step: 0.01, value: Math.log10(Math.abs(x)), aria: "x의 크기(log₁₀|x|)",
    onInput: (v) => { x = (x < 0 || Object.is(x, -0) ? -1 : 1) * 10 ** v; pre.select(null); update(); },
  });
  pm.row.style.gridTemplateColumns = "5.2em minmax(0, 1fr) 3.6em";
  const grid = el("div", "param-grid");
  grid.append(pm.row);
  const tools = el("div", "wframe-row");
  const neg = el("label", "toggle");
  neg.innerHTML = `<input type="checkbox" /> 음수`;
  const negIn = neg.querySelector("input")!;
  negIn.addEventListener("change", () => { x = -x; pre.select(null); update(); });
  tools.append(neg);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);

  function sync() {
    if (Number.isFinite(x) && x !== 0) pm.set(Math.max(-9, Math.min(8, Math.round(Math.log10(Math.abs(x)) * 100) / 100)));
    negIn.checked = x < 0 || Object.is(x, -0);
  }
  function update() {
    ro.set("x", `x = ${show(x)}`);
    for (const F of FMTS) {
      const r = roundTo(x, F);
      const U = 2 ** -(F.m + 1);
      if (Number.isNaN(x)) ro.set(F.key, `${F.name} NaN`, "hot");
      else if (!Number.isFinite(r) && Number.isFinite(x)) ro.set(F.key, `${F.name} 넘침 → inf`, "hot");
      else if (r === 0 && x !== 0) ro.set(F.key, `${F.name} 언더플로 → 0`, "hot");
      else if (!Number.isFinite(x) || x === 0) ro.set(F.key, `${F.name} 정확`, "ok");
      else {
        const rel = Math.abs(r - x) / Math.abs(x);
        const sub = Math.abs(r) < 2 ** emin(F);
        ro.set(F.key, `${F.name} 상대오차 ${sci(rel, 1)}${sub ? " · 서브노멀" : ` ≤ u ${sci(U, 1)}`}`, sub ? "hot" : rel === 0 ? "ok" : "");
      }
    }
    pageState.set("x", x, root);
    draw();
  }

  // 비트 칸 누르기
  const at = (e: PointerEvent): [number, number] | null => {
    const rc = f.canvas.getBoundingClientRect();
    const px = e.clientX - rc.left, py = e.clientY - rc.top;
    const g = geom();
    for (let i = 0; i < FMTS.length; i++) {
      const b0 = bitBox(g, i, 0);
      if (py < b0.y - 3 || py > b0.y + b0.h + 3) continue;
      const j = Math.floor((px - g.pad) / g.cw);
      if (j >= 0 && j < 1 + FMTS[i].e + FMTS[i].m) return [i, j];
    }
    return null;
  };
  f.canvas.addEventListener("pointermove", (e) => {
    const h = at(e);
    f.canvas.style.cursor = h ? "pointer" : "default";
    if (String(h) !== String(hover)) { hover = h; draw(); }
  });
  f.canvas.addEventListener("pointerleave", () => { hover = null; draw(); });
  f.canvas.addEventListener("click", (e) => {
    const h = at(e as PointerEvent);
    if (!h) return;
    const F = FMTS[h[0]];
    const bits = encode(roundTo(x, F), F);
    bits[h[1]] ^= 1;
    x = decode(bits, F);
    pre.select(null);
    sync();
    update();
  });

  f.onResize(draw);
  sync();
  update();
}
