// MI 추정 편향 (data/widgets.toml: mi-histogram-bias). 14.2장의 히스토그램(plug-in) 추정량 실험을 그대로 쓴다.
// 상관 0.6인 가우시안 쌍(참 MI = −½ log(1 − 0.36) = 0.223 nat) 또는 독립인 쌍(참 MI 0)에서 N개를 뽑아,
// 축마다 B개 bin의 결합 히스토그램으로 MI를 추정한다(np.histogram2d처럼 각 축의 최솟값~최댓값을 B등분).
// 왼쪽: 첫 번째 표본과 bin 격자(칸의 진하기 = 개수). 오른쪽: bin 수에 따른 20번 반복 평균 추정값과 참값.
// bin이 굵으면 양자화로 아래로, 가늘면 빈 칸과 우연한 몰림으로 위로 편향된다. 근사 편향은 (B − 1)²/(2N)이다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

const RHO = 0.6, REPS = 20;
export const NS = [50, 100, 200, 500, 1000, 2000, 5000];
const NMAX = NS[NS.length - 1];
export const BMIN = 2, BMAX = 50;
type Mode = "corr" | "indep";
export const trueMI = (mode: Mode) => (mode === "corr" ? -0.5 * Math.log(1 - RHO * RHO) : 0);

/** 반복마다 (x, e) 표준정규 표본. 상관 쌍은 y = ρx + √(1−ρ²) e, 독립 쌍은 (x, e)다(장의 코드와 같다). */
export const SAMPLES = Array.from({ length: REPS }, (_, k) => {
  const r = rng(9001 + k);
  const x = new Float64Array(NMAX), e = new Float64Array(NMAX);
  for (let i = 0; i < NMAX; i++) { x[i] = r.normal(); e[i] = r.normal(); }
  return { x, e };
});
export function pair(rep: number, n: number, mode: Mode): [Float64Array, Float64Array] {
  const { x, e } = SAMPLES[rep];
  const xs = x.subarray(0, n);
  if (mode === "indep") return [xs, e.subarray(0, n)];
  const s = Math.sqrt(1 - RHO * RHO);
  return [xs, Float64Array.from({ length: n }, (_, i) => RHO * x[i] + s * e[i])];
}

const range = (a: Float64Array): [number, number] => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
const binOf = (v: number, lo: number, hi: number, B: number) => Math.min(B - 1, Math.floor(((v - lo) / (hi - lo)) * B));
/** 결합 히스토그램의 칸 개수, (B, B) 행 우선 */
export function hist2d(x: Float64Array, y: Float64Array, B: number) {
  const [x0, x1] = range(x), [y0, y1] = range(y);
  const c = new Int32Array(B * B);
  for (let i = 0; i < x.length; i++) c[binOf(x[i], x0, x1, B) * B + binOf(y[i], y0, y1, B)]++;
  return { c, x0, x1, y0, y1 };
}
/** plug-in MI = H(X) + H(Y) − H(X, Y) [nat] */
export function miHist(x: Float64Array, y: Float64Array, B: number): number {
  const { c } = hist2d(x, y, B);
  const n = x.length, px = new Float64Array(B), py = new Float64Array(B);
  let hxy = 0;
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    const v = c[i * B + j];
    if (!v) continue;
    px[i] += v; py[j] += v;
    const p = v / n; hxy -= p * Math.log(p);
  }
  const H = (a: Float64Array) => a.reduce((s, v) => (v ? s - (v / n) * Math.log(v / n) : s), 0);
  return H(px) + H(py) - hxy;
}
/** bin 수 BMIN..BMAX마다 REPS번 반복한 평균 추정값 */
export function meanCurve(n: number, mode: Mode): number[] {
  const pairs = Array.from({ length: REPS }, (_, k) => pair(k, n, mode));
  return Array.from({ length: BMAX - BMIN + 1 }, (_, i) => pairs.reduce((s, [x, y]) => s + miHist(x, y, BMIN + i), 0) / REPS);
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "MI 추정 편향", "manipulate", narrow ? 1.25 : 0.56, "표본 산점도와 bin 격자, 그리고 bin 수에 따른 히스토그램 MI 추정값의 평균과 참값");
  let mode: Mode = "corr", ni = 2, B = 20;
  const cache = new Map<string, number[]>();
  const curveFor = (n: number, m: Mode) => { const k = `${m}${n}`; let c = cache.get(k); if (!c) { c = meanCurve(n, m); cache.set(k, c); } return c; };
  let curve = curveFor(NS[ni], mode);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const n = NS[ni];
    // 배치: 넓으면 좌우, 좁으면 위아래
    const side = narrow ? Math.min(W - 32, H * 0.5 - 20) : Math.min(H - 28, W * 0.44);
    const sx = narrow ? (W - side) / 2 : 16, sy = 14;
    const [x, y] = pair(0, n, mode);
    const h = hist2d(x, y, B);
    const PX = (v: number) => sx + ((v - h.x0) / (h.x1 - h.x0)) * side;
    const PY = (v: number) => sy + side - ((v - h.y0) / (h.y1 - h.y0)) * side;
    let cmax = 0; h.c.forEach((v) => { if (v > cmax) cmax = v; });
    const cw = side / B;
    ctx.fillStyle = C.ink2;
    for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
      const v = h.c[i * B + j];
      if (!v) continue;
      ctx.globalAlpha = 0.15 + 0.75 * (v / cmax);
      ctx.fillRect(sx + i * cw, sy + side - (j + 1) * cw, cw, cw);
    }
    ctx.globalAlpha = 1;
    if (B <= 60) {
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= B; k++) { ctx.moveTo(sx + k * cw, sy); ctx.lineTo(sx + k * cw, sy + side); ctx.moveTo(sx, sy + k * cw); ctx.lineTo(sx + side, sy + k * cw); }
      ctx.stroke();
    }
    ctx.fillStyle = C.ink;
    const r = n > 1000 ? 1 : 1.6;
    for (let i = 0; i < n; i++) ctx.fillRect(PX(x[i]) - r / 2, PY(y[i]) - r / 2, r, r);
    // 오른쪽(또는 아래): bin 수에 따른 평균 추정값
    const gx0 = narrow ? 44 : sx + side + 54, gx1 = W - 16;
    const gy0 = narrow ? sy + side + 26 : 20, gy1 = H - 26;
    const vmax = Math.max(0.5, Math.min(3, Math.max(...curve) * 1.08), trueMI("corr") * 1.6);
    const GX = (b: number) => gx0 + ((b - BMIN) / (BMAX - BMIN)) * (gx1 - gx0);
    const GY = (v: number) => gy1 - (Math.min(v, vmax) / vmax) * (gy1 - gy0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(gx0, gy0); ctx.lineTo(gx0, gy1); ctx.lineTo(gx1, gy1); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const b of [2, 10, 20, 30, 40, 50]) { const s = String(b); ctx.fillText(s, GX(b) - ctx.measureText(s).width / 2, gy1 + 13); }
    const step = vmax > 2 ? 1 : vmax > 1 ? 0.5 : 0.25;
    for (let v = 0; v <= vmax + 1e-9; v += step) { const s = v.toFixed(2); ctx.fillText(s, gx0 - 6 - ctx.measureText(s).width, GY(v) + 4); }
    ctx.fillText("bin 수 B", gx1 - ctx.measureText("bin 수 B").width, gy1 - 6);
    // 참값
    const tv = trueMI(mode);
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(gx0, GY(tv)); ctx.lineTo(gx1, GY(tv)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.good; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText(`참값 ${tv.toFixed(3)}`, gx0 + 6, Math.max(gy0 + 10, GY(tv) - 6));
    // 평균 추정값 곡선
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); curve.forEach((v, i) => (i ? ctx.lineTo(GX(BMIN + i), GY(v)) : ctx.moveTo(GX(BMIN + i), GY(v)))); ctx.stroke();
    // 현재 B
    const cv = curve[B - BMIN];
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(GX(B), gy0); ctx.lineTo(GX(B), gy1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(GX(B), GY(cv), 5, 0, 7); ctx.fill();
  }

  function update(recompute: boolean) {
    const n = NS[ni];
    if (recompute) curve = curveFor(n, mode);
    const [x, y] = pair(0, n, mode);
    const one = miHist(x, y, B), avg = curve[B - BMIN], tv = trueMI(mode);
    const h = hist2d(x, y, B);
    const empty = h.c.reduce((s, v) => s + (v ? 0 : 1), 0) / (B * B);
    ro.set("avg", `20번 평균 ${avg.toFixed(3)} nat`, "hot");
    ro.set("true", `참값 ${tv.toFixed(3)}`, Math.abs(avg - tv) < 0.02 ? "ok" : "");
    ro.set("one", `이 표본 ${one.toFixed(3)}`);
    ro.set("bias", `(B−1)²/2N ${(((B - 1) ** 2) / (2 * n)).toFixed(3)}`);
    ro.set("cells", `빈 칸 ${Math.round(empty * 100)}%`);
    draw();
  }

  const modeBtns = presets<Mode>([{ key: "corr", label: "상관 ρ = 0.6" }, { key: "indep", label: "독립" }], (k) => { mode = k; update(true); }, mode);
  modeBtns.row.classList.add("segmented");
  let raf = 0;
  const later = (recompute: boolean) => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => update(recompute)); };
  const pb = param({ label: "B", min: BMIN, max: BMAX, step: 1, value: B, fmt: (v) => `${v}`, aria: "축마다 bin 수", onInput: (v) => { B = v; later(false); } });
  const pn = param({ label: "N", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => `${NS[v]}`, aria: "표본 수", onInput: (v) => { ni = v; later(true); } });
  pn.input.setAttribute("aria-valuetext", String(NS[ni]));
  pn.input.addEventListener("input", () => pn.input.setAttribute("aria-valuetext", String(NS[ni])));
  const grid = el("div", "param-grid");
  grid.append(pb.row, pn.row);
  const ro = readouts();
  f.panel.append(modeBtns.row, grid, ro.row);
  f.onResize(draw);
  update(false);
}
