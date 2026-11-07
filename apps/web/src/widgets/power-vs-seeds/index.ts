// 검정력과 seed 수 (data/widgets.toml: power-vs-seeds).
// 23.5장 "검정력"·"블록화": 실행 간 표준편차 σ = 0.02, 같은 분할에서 두 방법 점수의 상관 ρ, 검출하려는 개선 δ, 유의수준 0.05 양측.
// 짝지음: 차이의 표준편차 σ_d = σ√(2(1−ρ)), 자유도 n−1, 비중심도 δ/(σ_d/√n).
// 짝짓지 않음: 자유도 2n−2, 비중심도 δ/(σ√(2/n)).
// 검정력은 비중심 t 분포로 정확히 계산한다: T = (Z + nc)/S, S = √(V/df), V ~ χ²(df). P(|T| > t*) = ∫ f(s)[Φ(nc − t*s) + Φ(−nc − t*s)] ds.
import { frame, param, readouts, el, C, alpha } from "../frame";

export const SIGMA = 0.02;

/** 표준정규 누적분포 Φ(x). erfc는 Numerical Recipes의 Chebyshev 근사(상대오차 < 1.2e-7). */
export function Phi(x: number) {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r / 2 : r / 2;
}

const NQ = 801;
const quad = new Map<number, { s: Float64Array; w: Float64Array }>();
/** S = √(V/df)의 밀도를 격자 위에서 정규화한 Simpson 가중치 */
function grid(df: number) {
  let g = quad.get(df);
  if (g) return g;
  const sd = 1 / Math.sqrt(2 * df);
  const lo = Math.max(0, 1 - 9 * sd), hi = 1 + 9 * sd + (df < 6 ? 8 : 0);
  const h = (hi - lo) / (NQ - 1), s = new Float64Array(NQ), w = new Float64Array(NQ);
  const mode = Math.sqrt(Math.max(df - 1, 0) / df);
  const logf = (x: number) => (df === 1 ? 0 : x > 0 ? (df - 1) * Math.log(x) : -Infinity) - (df * x * x) / 2;
  const ref = logf(mode || 0);
  let z = 0;
  for (let i = 0; i < NQ; i++) {
    s[i] = lo + i * h;
    const simpson = i === 0 || i === NQ - 1 ? 1 : i % 2 ? 4 : 2;
    w[i] = Math.exp(logf(s[i]) - ref) * simpson;
    z += w[i];
  }
  for (let i = 0; i < NQ; i++) w[i] /= z;
  g = { s, w };
  quad.set(df, g);
  return g;
}
const tail = (df: number, c: number, nc: number) => { const { s, w } = grid(df); let p = 0; for (let i = 0; i < NQ; i++) p += w[i] * (Phi(nc - c * s[i]) + Phi(-nc - c * s[i])); return p; };
const tcache = new Map<number, number>();
/** t 분포의 0.975 분위수(양측 0.05의 임계값), 이분법 */
export function tcrit(df: number) {
  let c = tcache.get(df);
  if (c !== undefined) return c;
  let a = 1.95, b = 12.8;                                   // df = 1의 12.706과 df → ∞의 1.960을 감싼다
  for (let it = 0; it < 34; it++) { const m = (a + b) / 2; if (tail(df, m, 0) > 0.05) a = m; else b = m; }
  c = (a + b) / 2; tcache.set(df, c);
  return c;
}
export function power(n: number, paired: boolean, delta: number, rho: number, sigma = SIGMA) {
  const df = paired ? n - 1 : 2 * n - 2;
  const nc = paired ? delta / ((sigma * Math.sqrt(2 * (1 - rho))) / Math.sqrt(n)) : delta / (sigma * Math.sqrt(2 / n));
  return tail(df, tcrit(df), nc);
}
/** 검정력 0.8에 필요한 가장 작은 n(방법당), 상한 nmax */
export function nNeeded(paired: boolean, delta: number, rho: number, nmax = 2000) {
  if (power(nmax, paired, delta, rho) < 0.8) return Infinity;
  let a = 2, b = nmax;
  if (power(a, paired, delta, rho) >= 0.8) return a;
  while (b - a > 1) { const m = (a + b) >> 1; if (power(m, paired, delta, rho) >= 0.8) b = m; else a = m; }
  return b;
}

const NMAX = 80;
export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "검정력과 seed 수", "manipulate", narrow ? 1.0 : 0.6, "실행 수에 따른 짝지은 설계와 짝짓지 않은 설계의 검정력 곡선, 그리고 추정한 차이의 분포");
  let delta = 0.01, n = 10, rho = 0.8;
  let curves: { p: number[]; u: number[] } = { p: [], u: [] };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 38, R = 14, top = 24, split = H * 0.62, bottom = H - 26;
    const X = (k: number) => L + ((k - 2) / (NMAX - 2)) * (W - L - R);
    const Y = (p: number) => split - 18 - p * (split - 18 - top);
    ctx.textBaseline = "alphabetic";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, split - 18); ctx.lineTo(W - R, split - 18); ctx.moveTo(L, top); ctx.lineTo(L, split - 18); ctx.stroke();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const p of [0, 0.5, 1]) ctx.fillText(p.toFixed(1), 6, Y(p) + 4);
    for (const k of narrow ? [2, 20, 40, 60, 80] : [2, 10, 20, 30, 40, 50, 60, 70, 80]) { const s = String(k); ctx.fillText(s, X(k) - ctx.measureText(s).width / 2, split - 4); }
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.good; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(L, Y(0.8)); ctx.lineTo(W - R, Y(0.8)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.good; ctx.fillText("0.8", 6, Y(0.8) + 4);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("검정력 — 방법당 실행(seed) 수 n →", L + 4, top - 8);
    const line = (v: number[], col: string, w: number, dash: number[]) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); v.forEach((p, i) => (i ? ctx.lineTo(X(i + 2), Y(p)) : ctx.moveTo(X(i + 2), Y(p)))); ctx.stroke(); ctx.setLineDash([]);
    };
    line(curves.u, C.mid, 1.8, [6, 4]);
    line(curves.p, C.ink2, 2.4, []);
    // 현재 n
    const pp = curves.p[n - 2], pu = curves.u[n - 2];
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(n), top); ctx.lineTo(X(n), split - 18); ctx.stroke();
    ctx.fillStyle = C.edge; for (const p of [pp, pu]) { ctx.beginPath(); ctx.arc(X(n), Y(p), 5, 0, 7); ctx.fill(); }
    // 범례
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const lx = W - R - (narrow ? 150 : 190), ly = Y(0.3);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.fillText("짝지음", lx + 24, ly + 4);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(lx, ly + 18); ctx.lineTo(lx + 18, ly + 18); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("짝짓지 않음", lx + 24, ly + 22);

    // 아래: 추정한 차이 δ̂의 분포(정규, 표준오차 = 짝지음 σ_d/√n, 짝짓지 않음 σ√(2/n))
    const seP = (SIGMA * Math.sqrt(2 * (1 - rho))) / Math.sqrt(n), seU = SIGMA * Math.sqrt(2 / n);
    const span = Math.max(delta + 3.2 * seU, 3.2 * seU, 0.01);
    const x0 = -span * 0.75, x1 = span;
    const top2 = split + 20;
    const X2 = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const dens = (v: number, se: number) => Math.exp(-0.5 * ((v - delta) / se) ** 2) / se;
    const peak = 1 / Math.min(seP, seU);
    const Y2 = (d: number) => bottom - (d / peak) * (bottom - top2 - 6);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, bottom); ctx.lineTo(W - R, bottom); ctx.stroke();
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("추정한 차이 δ̂의 흔들림", L + 4, top2 - 4);
    const curve = (se: number, col: string, fill: string, dash: number[]) => {
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) { const v = x0 + ((x1 - x0) * i) / 160; const x = X2(v), y = Y2(dens(v, se)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      if (fill) { ctx.lineTo(X2(x1), bottom); ctx.lineTo(X2(x0), bottom); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }
      ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
    };
    curve(seU, C.mid, alpha(C.mid, 0.12), [6, 4]);
    curve(seP, C.ink2, alpha(C.ink2, 0.2), []);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X2(0), top2 + 4); ctx.lineTo(X2(0), bottom + 4); ctx.stroke();
    ctx.fillText("0", X2(0) - 3, bottom + 16);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X2(delta), top2 + 4); ctx.lineTo(X2(delta), bottom + 4); ctx.stroke();
    ctx.fillStyle = C.edge; const dl = `δ ${delta.toFixed(3)}`;
    ctx.fillText(dl, Math.min(X2(delta) + 4, W - R - ctx.measureText(dl).width), bottom + 16);
  }

  function update(recurve = true) {
    if (recurve) {
      const ks = Array.from({ length: NMAX - 1 }, (_, i) => i + 2);
      curves = { p: ks.map((k) => power(k, true, delta, rho)), u: ks.map((k) => power(k, false, delta, rho)) };
    }
    const pp = curves.p[n - 2], pu = curves.u[n - 2];
    ro.set("pow", `n = ${n}: 짝지음 ${pp.toFixed(2)} · 짝짓지 않음 ${pu.toFixed(2)}`, "hot");
    ro.set("sd", `σ_d = σ√(2(1−ρ)) = ${(SIGMA * Math.sqrt(2 * (1 - rho))).toFixed(4)}`);
    if (recurve) {
      const fmt = (v: number) => (Number.isFinite(v) ? String(v) : "> 2000");
      ro.set("need", `0.8에 필요한 n: 짝지음 ${fmt(nNeeded(true, delta, rho))} · 짝짓지 않음 ${fmt(nNeeded(false, delta, rho))}`, "ok");
    }
    draw();
  }

  const pd = param({ label: "δ", min: 0.002, max: 0.03, step: 0.001, value: delta, fmt: (v) => v.toFixed(3), aria: "검출하려는 개선 δ (정확도)", onInput: (v) => { delta = v; update(); } });
  const pn = param({ label: "n", min: 2, max: NMAX, step: 1, value: n, fmt: (v) => String(v), aria: "방법당 실행(seed) 수 n", onInput: (v) => { n = v; update(false); } });
  const pr = param({ label: "ρ", min: 0, max: 0.99, step: 0.01, value: rho, aria: "같은 분할에서 두 방법 점수의 상관 ρ", onInput: (v) => { rho = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pd.row, pn.row, pr.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
