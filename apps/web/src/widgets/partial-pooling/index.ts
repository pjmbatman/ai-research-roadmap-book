// 부분 풀링과 수축 (data/widgets.toml: partial-pooling).
// 13.3장의 예: 가상 배터리 셀 J = 8개의 열화율 [%/100 cycles]. 참값 θ_j = μ + τ* z_j (μ = 2.0, τ* = 0.3), 측정 y_ij = θ_j + σ ε_ij (σ = 0.6).
// 셀 평균 ȳ_j의 표준오차 s_j = σ/√n_j, 수축 계수 B_j = s_j² / (s_j² + τ²), 부분 풀링 추정값 B_j μ̂ + (1 − B_j) ȳ_j.
// 모형의 τ는 슬라이더로 정하고, μ̂(τ)는 정밀도 가중평균 Σ ȳ_j/(τ² + s_j²) / Σ 1/(τ² + s_j²)이다(τ → 0이면 전체 평균, 곧 complete pooling).
// 주변우도 ȳ_j | μ, τ ~ N(μ, τ² + s_j²)를 최대로 하는 τ̂(empirical Bayes)도 함께 표시한다. 잡음은 고정 시드.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const J = 8, MU = 2.0, TAU_TRUE = 0.3, SIGMA = 0.6, NMAX = 40;
const RAW = (() => {
  const r = rng(1975);
  const z = Array.from({ length: J }, () => r.normal());
  const eps = Array.from({ length: J }, () => Array.from({ length: NMAX }, () => r.normal()));
  return { z, eps };
})();
export const THETA = RAW.z.map((z) => MU + TAU_TRUE * z);

export const LAYOUTS = {
  book: { label: "1~40번 (본문)", n: [1, 2, 3, 5, 8, 12, 20, 40] },
  few: { label: "모두 2번", n: [2, 2, 2, 2, 2, 2, 2, 2] },
  many: { label: "모두 20번", n: [20, 20, 20, 20, 20, 20, 20, 20] },
} as const;
export type LK = keyof typeof LAYOUTS;

export function groupData(nj: readonly number[]) {
  const ybar = nj.map((n, j) => { let s = 0; for (let i = 0; i < n; i++) s += THETA[j] + SIGMA * RAW.eps[j][i]; return s / n; });
  const se = nj.map((n) => SIGMA / Math.sqrt(n));
  const total = nj.reduce((s, n) => s + n, 0);
  const grand = ybar.reduce((s, y, j) => s + y * nj[j], 0) / total;
  return { ybar, se, grand };
}
export function pool(ybar: number[], se: number[], tau: number) {
  const v = se.map((s) => tau * tau + s * s);
  const mu = ybar.reduce((s, y, j) => s + y / v[j], 0) / v.reduce((s, x) => s + 1 / x, 0);
  const B = se.map((s) => (s * s) / (s * s + tau * tau));
  const est = ybar.map((y, j) => B[j] * mu + (1 - B[j]) * y);
  const ll = -0.5 * ybar.reduce((s, y, j) => s + Math.log(v[j]) + (y - mu) ** 2 / v[j], 0);
  return { mu, B, est, ll };
}
/** 주변우도를 최대로 하는 τ̂ (격자 0.005~1.5, 601점) */
export function tauHatOf(ybar: number[], se: number[]) {
  let best = -Infinity, arg = 0;
  for (let i = 0; i <= 600; i++) { const t = 0.005 + (1.5 - 0.005) * (i / 600), ll = pool(ybar, se, t).ll; if (ll > best) { best = ll; arg = t; } }
  return arg;
}
export const rmse = (est: number[]) => Math.sqrt(est.reduce((s, e, j) => s + (e - THETA[j]) ** 2, 0) / J);

const LT0 = -2, LT1 = 0.5; // log10 τ 범위
const TAUS = Array.from({ length: 121 }, (_, i) => Math.pow(10, LT0 + ((LT1 - LT0) * i) / 120));
const A0 = 0.6, A1 = 3.4;

export function mount(root: HTMLElement) {
  const f = frame(root, "부분 풀링과 수축", "manipulate", 0.82, "셀 8개의 따로 추정값에서 부분 풀링 추정값으로의 이동, 참값, 그리고 τ에 따른 RMSE 곡선");
  let layout: LK = "book", lt = Math.log10(0.3);
  let gd = groupData(LAYOUTS[layout].n);
  let curve: number[] = [], tauHat = 0;

  function recompute() {
    gd = groupData(LAYOUTS[layout].n);
    curve = TAUS.map((t) => rmse(pool(gd.ybar, gd.se, t).est));
    tauHat = tauHatOf(gd.ybar, gd.se);
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const tau = Math.pow(10, lt), P = pool(gd.ybar, gd.se, tau), nj: readonly number[] = LAYOUTS[layout].n;
    const PL = 46, PR = 12, top = 38, base1 = Math.round(H * 0.6);
    const X = (v: number) => PL + ((v - A0) / (A1 - A0)) * (W - PL - PR);
    const rowH = (base1 - top) / J;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (let v = 1; v <= 3; v += 0.5) { ctx.beginPath(); ctx.moveTo(X(v), top - 6); ctx.lineTo(X(v), base1); ctx.stroke(); const s = v.toFixed(1); ctx.fillText(s, X(v) - ctx.measureText(s).width / 2, base1 + 14); }
    // 모집단 평균 μ̂
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(P.mu), top - 6); ctx.lineTo(X(P.mu), base1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.mid;
    const ml = `평균 ${P.mu.toFixed(2)}`;
    ctx.fillText(ml, Math.min(W - PR - ctx.measureText(ml).width, X(P.mu) + 4), top - 8);
    ctx.fillText("가상 셀 8개", 4, top - 8);
    for (let j = 0; j < J; j++) {
      const yy = top + (j + 0.5) * rowH;
      ctx.fillStyle = C.mid; ctx.fillText(`n=${nj[j]}`, 4, yy + 4);
      // ±1 SE
      ctx.strokeStyle = alpha(C.ink2, 0.35); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(Math.max(A0, gd.ybar[j] - gd.se[j])), yy); ctx.lineTo(X(Math.min(A1, gd.ybar[j] + gd.se[j])), yy); ctx.stroke();
      // 참값
      ctx.strokeStyle = C.good; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(THETA[j]), yy - rowH * 0.36); ctx.lineTo(X(THETA[j]), yy + rowH * 0.36); ctx.stroke();
      // 이동 화살표: 따로 추정 → 부분 풀링
      const x0 = X(gd.ybar[j]), x1 = X(P.est[j]);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
      ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(x0, yy, 4, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.edge;
      ctx.beginPath(); ctx.arc(x1, yy, 4.5, 0, 7); ctx.fill();
    }
    // 범례
    const items: [string, string, "o" | "dot" | "tick"][] = [["따로", C.ink2, "o"], ["부분 풀링", C.edge, "dot"], ["참값", C.good, "tick"]];
    let lx = Math.max(PL, 90);
    for (const [name, col, kind] of items) {
      const cy = 10;
      if (kind === "o") { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(lx + 4, cy - 3, 3.5, 0, 7); ctx.stroke(); }
      else if (kind === "dot") { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(lx + 4, cy - 3, 4, 0, 7); ctx.fill(); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx + 4, cy - 9); ctx.lineTo(lx + 4, cy + 3); ctx.stroke(); }
      ctx.fillStyle = C.mid; ctx.fillText(name, lx + 12, cy + 1);
      lx += 24 + ctx.measureText(name).width;
    }
    // 아래: τ에 따른 RMSE
    const t2 = base1 + 34, b2 = H - 22;
    const noPool = rmse(gd.ybar), complete = rmse(gd.ybar.map(() => gd.grand));
    const vmax = Math.max(noPool, complete, ...curve) * 1.08, vmin = 0;
    const XT = (l: number) => PL + ((l - LT0) / (LT1 - LT0)) * (W - PL - PR);
    const Y2 = (v: number) => b2 - ((v - vmin) / (vmax - vmin)) * (b2 - t2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, b2); ctx.lineTo(W - PR, b2); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (const t of [0.01, 0.03, 0.1, 0.3, 1, 3]) { const s = String(t); const x = XT(Math.log10(t)); ctx.fillText(s, Math.min(W - PR - ctx.measureText(s).width, x - ctx.measureText(s).width / 2), b2 + 14); }
    ctx.fillText("RMSE", 4, t2 + 4);
    ctx.fillText("τ", 4, b2 + 14);
    const hline = (v: number, col: string) => { ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(PL, Y2(v)); ctx.lineTo(W - PR, Y2(v)); ctx.stroke(); ctx.setLineDash([]); };
    hline(noPool, C.ink2); hline(complete, C.mid);
    ctx.fillStyle = C.ink2; ctx.fillText("따로", W - PR - ctx.measureText("따로").width, Y2(noPool) - 4);
    ctx.fillStyle = C.mid; ctx.fillText("완전", PL + 2, Y2(complete) - 4);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    ctx.beginPath(); curve.forEach((v, i) => { const x = XT(LT0 + ((LT1 - LT0) * i) / 120); i ? ctx.lineTo(x, Y2(v)) : ctx.moveTo(x, Y2(v)); }); ctx.stroke();
    // 참 τ, 주변우도 최대 τ̂, 현재 τ
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(XT(Math.log10(TAU_TRUE)), t2); ctx.lineTo(XT(Math.log10(TAU_TRUE)), b2); ctx.stroke();
    ctx.fillStyle = C.ink;
    const xh = XT(Math.log10(Math.max(tauHat, Math.pow(10, LT0))));
    ctx.beginPath(); ctx.moveTo(xh, b2 - 1); ctx.lineTo(xh - 5, b2 - 9); ctx.lineTo(xh + 5, b2 - 9); ctx.closePath(); ctx.fill();
    const cur = rmse(P.est);
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(XT(lt), Y2(cur), 5, 0, 7); ctx.fill();
  }

  let raf = 0;
  function update(full = false) {
    if (full) recompute();
    const tau = Math.pow(10, lt), P = pool(gd.ybar, gd.se, tau), nj: readonly number[] = LAYOUTS[layout].n;
    const noPool = rmse(gd.ybar), complete = rmse(gd.ybar.map(() => gd.grand)), part = rmse(P.est);
    ro.set("rmse", `RMSE 부분 풀링 ${part.toFixed(3)}`, part < Math.min(noPool, complete) ? "ok" : "");
    ro.set("ref", `따로 ${noPool.toFixed(3)} · 완전 풀링 ${complete.toFixed(3)}`);
    const iMin = nj.indexOf(Math.min(...nj)), iMax = nj.lastIndexOf(Math.max(...nj));
    ro.set("B", iMin === iMax || nj[iMin] === nj[iMax] ? `수축 B = ${P.B[0].toFixed(2)} (모든 셀)` : `수축 B: n=${nj[iMin]} ${P.B[iMin].toFixed(2)} · n=${nj[iMax]} ${P.B[iMax].toFixed(2)}`, "hot");
    ro.set("tau", `▲ 주변우도 최대 τ ${tauHat.toFixed(3)} (참 ${TAU_TRUE})`);
    pageState.set("pooling", { tau, layout }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(LAYOUTS) as LK[]).map((k) => ({ key: k, label: LAYOUTS[k].label })), (k) => { layout = k; update(true); }, layout);
  const pt = param({ label: "τ", min: LT0, max: LT1, step: 0.01, value: lt, fmt: (v) => { const t = Math.pow(10, v); return t < 0.1 ? t.toFixed(3) : t.toFixed(2); }, aria: "모형의 그룹 간 표준편차 τ (로그 눈금)", onInput: (v) => { lt = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update(true);
}
