// 잔차 진단 (data/widgets.toml: residual-diagnostics).
// 12.6장의 용량 열화 예: 사이클 c = 0, 5, …, 500 (n = 101)에서 용량 [Ah]을 재고 최소제곱으로 직선(또는 2차)을 맞춘다.
// 데이터 프리셋은 본문의 식을 따른다(잡음 z는 고정 시드).
//   정상: 2.5 − 0.001c + 0.01z           곡률: 2.5 − 0.0005c − 1.5×10⁻⁶c² + 0.01z
//   이분산: 2.5 − 0.001c + (0.005 + 0.00006c)z  자기상관: 2.5 − 0.001c + e_t, e_t = 0.7e_{t−1} + 0.01z_t
//   영향점: 2.5 − 0.001c + 0.01z, 마지막 사이클에서 −0.15
// 진단: 잔차 vs 사이클, 잔차 퍼짐 비(뒤 1/3과 앞 1/3의 |잔차| 중앙값 비), 표준화 잔차의 정규 Q-Q, Cook's distance D_i = e_i²/(p σ̂²) · h_ii/(1 − h_ii)², Durbin–Watson.
import { frame, presets, readouts, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const CYC = Array.from({ length: 101 }, (_, i) => 5 * i);
const n = CYC.length;
const Zr = (() => { const r = rng(1984); return Array.from({ length: n }, () => r.normal()); })();

export const DATA = {
  ok: { label: "정상", y: () => CYC.map((c, i) => 2.5 - 0.001 * c + 0.01 * Zr[i]) },
  curve: { label: "곡률", y: () => CYC.map((c, i) => 2.5 - 0.0005 * c - 1.5e-6 * c * c + 0.01 * Zr[i]) },
  hetero: { label: "이분산", y: () => CYC.map((c, i) => 2.5 - 0.001 * c + (0.005 + 0.00006 * c) * Zr[i]) },
  ar: { label: "자기상관", y: () => { let e = 0; return CYC.map((c, i) => { e = i ? 0.7 * e + 0.01 * Zr[i] : 0; return 2.5 - 0.001 * c + e; }); } },
  infl: { label: "영향점", y: () => CYC.map((c, i) => 2.5 - 0.001 * c + 0.01 * Zr[i] - (i === n - 1 ? 0.15 : 0)) },
} as const;
export type DK = keyof typeof DATA;

/** p×p 대칭 양의 정부호 행렬의 역(가우스 소거) */
function inv(A: number[][]) {
  const p = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: p }, (_, j) => (i === j ? 1 : 0))]);
  for (let k = 0; k < p; k++) {
    let piv = k;
    for (let i = k + 1; i < p; i++) if (Math.abs(M[i][k]) > Math.abs(M[piv][k])) piv = i;
    [M[k], M[piv]] = [M[piv], M[k]];
    const d = M[k][k];
    for (let j = 0; j < 2 * p; j++) M[k][j] /= d;
    for (let i = 0; i < p; i++) if (i !== k) { const m = M[i][k]; for (let j = 0; j < 2 * p; j++) M[i][j] -= m * M[k][j]; }
  }
  return M.map((r) => r.slice(p));
}

/** OLS 적합과 진단 통계 (입력은 c/500로 척도를 맞춰 계산하고, 계수는 사이클 단위로 되돌린다) */
export function diagnose(y: number[], p: 2 | 3) {
  const X = CYC.map((c) => { const u = c / 500; return p === 2 ? [1, u] : [1, u, u * u]; });
  const XtX = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => X.reduce((s, r) => s + r[a] * r[b], 0)));
  const Xi = inv(XtX);
  const Xty = Array.from({ length: p }, (_, a) => X.reduce((s, r, i) => s + r[a] * y[i], 0));
  const beta = Xi.map((row) => row.reduce((s, v, b) => s + v * Xty[b], 0));
  const fit = X.map((r) => r.reduce((s, v, a) => s + v * beta[a], 0));
  const e = y.map((v, i) => v - fit[i]);
  const rss = e.reduce((s, v) => s + v * v, 0), s2 = rss / (n - p);
  const h = X.map((r) => r.reduce((s, va, a) => s + va * r.reduce((t, vb, b) => t + Xi[a][b] * vb, 0), 0));
  const cook = e.map((v, i) => ((v * v) / (p * s2)) * (h[i] / (1 - h[i]) ** 2));
  const std = e.map((v, i) => v / Math.sqrt(s2 * (1 - h[i])));
  let dwn = 0;
  for (let i = 1; i < n; i++) dwn += (e[i] - e[i - 1]) ** 2;
  const se = Xi.map((r, a) => Math.sqrt(s2 * r[a]));
  // 계수를 사이클 단위로: β_c = β_u / 500, β_cc = β_uu / 500²
  const slope = beta[1] / 500, slopeSE = se[1] / 500;
  const third = Math.floor(n / 3);
  // 퍼짐은 |잔차|의 중앙값으로 잰다(영향점 하나에 끌리지 않게)
  const sdOf = (a: number[]) => { const s = a.map(Math.abs).sort((u, v) => u - v), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  return {
    beta, fit, e, h, cook, std, s2, slope, slopeSE, dw: dwn / rss,
    quadT: p === 3 ? beta[2] / se[2] : NaN,
    spreadRatio: sdOf(e.slice(n - third)) / sdOf(e.slice(0, third)),
  };
}
/** 표준정규 분위수 (Acklam 근사) */
export function normInv(q: number) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const lo = 0.02425;
  if (q < lo) { const t = Math.sqrt(-2 * Math.log(q)); return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1); }
  if (q > 1 - lo) return -normInv(1 - q);
  const s = q - 0.5, r = s * s;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * s) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "잔차 진단", "manipulate", 0.86, "열화 데이터와 적합선, 잔차 대 사이클, 정규 Q-Q plot, Cook's distance의 네 그림");
  let key: DK = (attrs.data in DATA ? attrs.data : "curve") as DK;
  let p: 2 | 3 = 2;
  let y = DATA[key].y();
  let dg = diagnose(y, p);
  const QN = Array.from({ length: n }, (_, i) => normInv((i + 1 - 0.375) / (n + 0.25)));

  function panel(x0: number, y0: number, w: number, h: number, title: string) {
    const { ctx } = f;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(title, x0 + 6, y0 + 14);
  }
  const range = (a: number[], pad = 0.08) => { const lo = Math.min(...a), hi = Math.max(...a), d = hi - lo || 1; return [lo - d * pad, hi + d * pad]; };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const g = 8, pw = (W - g) / 2, ph = (H - g) / 2;
    const inner = (x0: number, y0: number) => ({ l: x0 + 8, r: x0 + pw - 8, t: y0 + 22, b: y0 + ph - 8 });
    const dot = (x: number, yy: number, col: string, rad = 2) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, yy, rad, 0, 7); ctx.fill(); };
    const imax = dg.cook.indexOf(Math.max(...dg.cook));
    // 1. 데이터와 적합
    {
      panel(0, 0, pw, ph, "용량 [Ah] vs 사이클");
      const q = inner(0, 0), [lo, hi] = range(y);
      const X = (c: number) => q.l + (c / 500) * (q.r - q.l), Y = (v: number) => q.b - ((v - lo) / (hi - lo)) * (q.b - q.t);
      y.forEach((v, i) => dot(X(CYC[i]), Y(v), i === imax && key === "infl" ? C.edge : alpha(C.ink2, 0.8)));
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
      ctx.beginPath(); dg.fit.forEach((v, i) => (i ? ctx.lineTo(X(CYC[i]), Y(v)) : ctx.moveTo(X(CYC[i]), Y(v)))); ctx.stroke();
    }
    // 2. 잔차 vs 사이클
    {
      panel(pw + g, 0, pw, ph, "잔차 vs 사이클");
      const q = inner(pw + g, 0), m = Math.max(...dg.e.map(Math.abs)) * 1.1;
      const X = (c: number) => q.l + (c / 500) * (q.r - q.l), Y = (v: number) => (q.t + q.b) / 2 - (v / m) * ((q.b - q.t) / 2);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(q.l, Y(0)); ctx.lineTo(q.r, Y(0)); ctx.stroke(); ctx.setLineDash([]);
      if (key === "ar") { ctx.strokeStyle = alpha(C.ink2, 0.5); ctx.lineWidth = 1; ctx.beginPath(); dg.e.forEach((v, i) => (i ? ctx.lineTo(X(CYC[i]), Y(v)) : ctx.moveTo(X(CYC[i]), Y(v)))); ctx.stroke(); }
      dg.e.forEach((v, i) => dot(X(CYC[i]), Y(v), i === imax && key === "infl" ? C.edge : alpha(C.ink2, 0.85)));
      // 11칸 이동 평균: 곡률이 보이게
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const a = Math.max(0, i - 5), b = Math.min(n, i + 6); let s = 0; for (let k = a; k < b; k++) s += dg.e[k]; const v = s / (b - a); i ? ctx.lineTo(X(CYC[i]), Y(v)) : ctx.moveTo(X(CYC[i]), Y(v)); }
      ctx.stroke();
    }
    // 3. 정규 Q-Q
    {
      panel(0, ph + g, pw, ph, "정규 Q-Q (표준화 잔차)");
      const q = inner(0, ph + g);
      const sorted = [...dg.std].sort((a, b) => a - b);
      const m = Math.max(3, ...sorted.map(Math.abs)) * 1.05;
      const X = (v: number) => (q.l + q.r) / 2 + (v / 3.2) * ((q.r - q.l) / 2), Y = (v: number) => (q.t + q.b) / 2 - (v / m) * ((q.b - q.t) / 2);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X(-3.2), Y(-3.2)); ctx.lineTo(X(3.2), Y(3.2)); ctx.stroke(); ctx.setLineDash([]);
      sorted.forEach((v, i) => dot(X(QN[i]), Y(v), Math.abs(v) > 3 ? C.edge : alpha(C.ink2, 0.85)));
    }
    // 4. Cook's distance
    {
      panel(pw + g, ph + g, pw, ph, "Cook's distance");
      const q = inner(pw + g, ph + g), m = Math.max(1.2, Math.max(...dg.cook) * 1.1);
      const X = (c: number) => q.l + (c / 500) * (q.r - q.l), Y = (v: number) => q.b - (v / m) * (q.b - q.t);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(q.l, Y(1)); ctx.lineTo(q.r, Y(1)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText("D = 1", q.l + 2, Y(1) - 4);
      ctx.lineWidth = 1.6;
      dg.cook.forEach((v, i) => { ctx.strokeStyle = v > 0.5 ? C.edge : alpha(C.ink2, 0.8); ctx.beginPath(); ctx.moveTo(X(CYC[i]), q.b); ctx.lineTo(X(CYC[i]), Y(v)); ctx.stroke(); });
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(q.l, q.b); ctx.lineTo(q.r, q.b); ctx.stroke();
    }
  }

  let raf = 0;
  function update() {
    y = DATA[key].y();
    dg = diagnose(y, p);
    const imax = dg.cook.indexOf(Math.max(...dg.cook));
    ro.set("fit", p === 2 ? `기울기 ${(dg.slope * 1000).toFixed(3)} ± ${(dg.slopeSE * 1000).toFixed(3)} mAh/cycle` : `2차 항 t = ${dg.quadT.toFixed(1)}`);
    ro.set("sd", `잔차 퍼짐 비(뒤/앞) ${dg.spreadRatio.toFixed(2)}`, dg.spreadRatio > 1.5 ? "hot" : "");
    ro.set("dw", `Durbin–Watson ${dg.dw.toFixed(2)}`, dg.dw < 1.5 ? "hot" : "");
    ro.set("cook", `최대 Cook's D ${dg.cook[imax].toFixed(2)} (c=${CYC[imax]}, h=${dg.h[imax].toFixed(3)})`, dg.cook[imax] > 0.5 ? "hot" : "");
    pageState.set("residuals", { data: key, terms: p }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(DATA) as DK[]).map((k) => ({ key: k, label: DATA[k].label })), (k) => { key = k; update(); }, key);
  const mdl = presets<"2" | "3">([{ key: "2", label: "직선" }, { key: "3", label: "2차 항 추가" }], (k) => { p = k === "2" ? 2 : 3; update(); }, "2");
  mdl.row.classList.add("segmented");
  const ro = readouts();
  f.panel.append(pre.row, mdl.row, ro.row);
  f.onResize(draw);
  update();
}
