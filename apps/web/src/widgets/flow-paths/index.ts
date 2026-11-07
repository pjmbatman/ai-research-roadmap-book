// flow matching 경로 (data/widgets.toml: flow-paths, 19.6장 "rectified flow와 최적 수송").
// 본문의 1차원 예: 잡음 p₀ = N(0, 1), 데이터 0.3·N(−2, 0.3²) + 0.7·N(2, 0.3²), 가우시안 경로 x_t = α_t x₁ + σ_t ε.
//   직선  α = t, σ = 1 − t            삼각  α = sin(πt/2), σ = cos(πt/2)
// 독립 짝의 참 주변 속도장 v*(x, t) = E[α̇ x₁ + σ̇ ε | x_t = x]는 성분마다 결합 가우시안이라 닫힌 꼴이다(본문 v_star의 일반화).
// reflow를 켜면 짝을 흐름이 정한 결정적 짝 (x₀, T(x₀))로 바꾼다. 1차원에서 이 짝은 단조 수송 T = F⁻¹∘Φ(최적 수송)이고
// 궤적이 서로 교차하지 않으므로 주변 속도가 조건부 속도와 같다. 학습 오차가 없는 참 속도장만 쓰므로, W₁은 순수한 solver 오차다.
// 기준 표본은 참 흐름의 끝점 T(x₀)이고(1차원에서 흐름 사상은 단조이므로 정확히 T), x₀는 N(0, 1)의 분위수 2000개다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";

const PI = [0.3, 0.7], M = [-2, 2], S = [0.3, 0.3];
export type Path = "line" | "trig";
export const coef = (p: Path, t: number) => p === "line"
  ? { a: t, s: 1 - t, da: 1, ds: -1 }
  : { a: Math.sin((Math.PI * t) / 2), s: Math.cos((Math.PI * t) / 2), da: (Math.PI / 2) * Math.cos((Math.PI * t) / 2), ds: -(Math.PI / 2) * Math.sin((Math.PI * t) / 2) };

/** 독립 짝의 참 주변 속도장 */
export function vStar(p: Path, x: number, t: number) {
  const { a, s, da, ds } = coef(p, t);
  let lw0 = 0, lw1 = 0, c0 = 0, c1 = 0;
  for (let k = 0; k < 2; k++) {
    const vr = a * a * S[k] * S[k] + s * s, d = x - a * M[k];
    const lw = Math.log(PI[k]) - (0.5 * d * d) / vr - 0.5 * Math.log(vr);
    const c = da * M[k] + ((da * a * S[k] * S[k] + ds * s) / vr) * d;
    if (k === 0) { lw0 = lw; c0 = c; } else { lw1 = lw; c1 = c; }
  }
  const m = Math.max(lw0, lw1), w0 = Math.exp(lw0 - m), w1 = Math.exp(lw1 - m);
  return (w0 * c0 + w1 * c1) / (w0 + w1);
}

// 표준정규 CDF(erfc, Numerical Recipes의 Chebyshev 근사, 상대오차 < 1.2e-7)와 그 역(Acklam 근사 + Newton 한 번)
function erfc(x: number) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
export const Phi = (x: number) => 0.5 * erfc(-x / Math.SQRT2);
export function PhiInv(p: number) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  let x: number;
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else { const q = p - 0.5, r = q * q; x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  const e = Phi(x) - p;
  return x - e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
}
const Fdata = (x: number) => PI[0] * Phi((x - M[0]) / S[0]) + PI[1] * Phi((x - M[1]) / S[1]);
/** 단조 수송 T(x₀) = F_data⁻¹(Φ(x₀)) (이분법) */
export function transport(x0: number) {
  const p = Phi(x0);
  let lo = -8, hi = 8;
  for (let i = 0; i < 60; i++) { const mid = 0.5 * (lo + hi); if (Fdata(mid) < p) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

// reflow 짝의 속도장: x = α T(x₀) + σ x₀를 표로 거꾸로 풀어 x₀를 찾고 v = α̇ T(x₀) + σ̇ x₀
const G = 4001, GX = Array.from({ length: G }, (_, i) => -6 + (12 * i) / (G - 1)), GT = GX.map(transport);
export function vReflow(p: Path, x: number, t: number) {
  const { a, s, da, ds } = coef(p, t);
  const g = (i: number) => a * GT[i] + s * GX[i];
  let lo = 0, hi = G - 1;
  if (x <= g(0)) hi = 1; else if (x >= g(G - 1)) lo = G - 2;
  else while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (g(mid) <= x) lo = mid; else hi = mid; }
  const r = (x - g(lo)) / (g(hi) - g(lo));
  const x0 = GX[lo] + r * (GX[hi] - GX[lo]), T = GT[lo] + r * (GT[hi] - GT[lo]);
  return da * T + ds * x0;
}

export const N0 = 2000;
export const X0S = Array.from({ length: N0 }, (_, i) => PhiInv((i + 0.5) / N0));
export const REF = X0S.map(transport);
export function euler(vf: (x: number, t: number) => number, xs: number[], n: number) {
  const out = xs.slice(), dt = 1 / n;
  for (let k = 0; k < n; k++) for (let i = 0; i < out.length; i++) out[i] += vf(out[i], k * dt) * dt;
  return out;
}
export function w1(a: number[], b: number[]) {
  const sa = a.slice().sort((p, q) => p - q), sb = b.slice().sort((p, q) => p - q);
  let s = 0;
  for (let i = 0; i < sa.length; i++) s += Math.abs(sa[i] - sb[i]);
  return s / sa.length;
}
/** CFM 손실의 바닥 E_t E_{x_t} Var[u | x_t] (독립 짝). E[u²] = α̇² E[x₁²] + σ̇², E_{x_t}[v*²]는 격자 적분 */
export function cfmFloor(p: Path) {
  const Ex1sq = PI[0] * (M[0] ** 2 + S[0] ** 2) + PI[1] * (M[1] ** 2 + S[1] ** 2);
  let tot = 0;
  const NT = 200, NX = 1200;
  for (let j = 0; j < NT; j++) {
    const t = (j + 0.5) / NT, { a, s, da, ds } = coef(p, t);
    let ev2 = 0;
    for (let i = 0; i < NX; i++) {
      const x = -7 + (14 * (i + 0.5)) / NX;
      let px = 0;
      for (let k = 0; k < 2; k++) { const vr = a * a * S[k] * S[k] + s * s; px += (PI[k] * Math.exp((-0.5 * (x - a * M[k]) ** 2) / vr)) / Math.sqrt(2 * Math.PI * vr); }
      ev2 += px * vStar(p, x, t) ** 2 * (14 / NX);
    }
    tot += (da * da * Ex1sq + ds * ds - ev2) / NT;
  }
  return tot;
}

export const STEPS = [1, 2, 4, 8, 16, 32, 64, 128];

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 440;
  const f = frame(root, "flow matching 경로", "manipulate", narrow ? 0.72 : 0.5,
    "시간에 따른 ODE 궤적, Euler 궤적, 조건부 경로와 끝점의 히스토그램");
  let path: Path = "line", reflow = false, si = STEPS.indexOf(4), showCond = true;
  const floors: Record<Path, number> = { line: cfmFloor("line"), trig: cfmFloor("trig") };
  const show = Array.from({ length: 21 }, (_, i) => X0S[Math.round(((i + 0.5) / 21) * N0 - 0.5)]);
  const r = rng(5);
  const pairs = Array.from({ length: 18 }, () => { const x0 = r.normal(); const k = r.u() < PI[0] ? 0 : 1; return [x0, M[k] + S[k] * r.normal()] as [number, number]; });
  let ends: number[] = [];
  const Y0 = -4.2, Y1 = 4.2;

  const vf = () => (reflow ? (x: number, t: number) => vReflow(path, x, t) : (x: number, t: number) => vStar(path, x, t));

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const histW = Math.max(70, W * 0.2);
    const g = { x: 54, y: 22, w: W - 54 - histW - 26, h: H - 50 };
    const hx = g.x + g.w + 16;
    const TX = (t: number) => g.x + t * g.w, XY = (x: number) => g.y + g.h - ((x - Y0) / (Y1 - Y0)) * g.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(g.x, g.y, g.w, g.h);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const x of [-4, -2, 0, 2, 4]) ctx.fillText(String(x), g.x - 20, XY(x) + 4);
    ctx.fillText("t = 0 잡음", g.x, g.y + g.h + 14);
    const lab = "t = 1 데이터"; ctx.fillText(lab, g.x + g.w - ctx.measureText(lab).width, g.y + g.h + 14);
    // 잡음 밀도(왼쪽 가장자리 바깥)
    ctx.strokeStyle = C.mid; ctx.beginPath();
    for (let i = 0; i <= 120; i++) { const x = Y0 + ((Y1 - Y0) * i) / 120; const px = g.x - 4 - 30 * Math.exp(-0.5 * x * x); if (i) ctx.lineTo(px, XY(x)); else ctx.moveTo(px, XY(x)); }
    ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(g.x, g.y, g.w, g.h); ctx.clip();
    const v = vf();
    // 조건부 경로(짝 하나의 경로): 독립 짝은 서로 교차하고, reflow 짝은 참 궤적과 같다
    if (showCond) {
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
      for (const [x0, x1i] of pairs) {
        const x1 = reflow ? transport(x0) : x1i;
        ctx.beginPath();
        for (let i = 0; i <= 40; i++) { const t = i / 40, c = coef(path, t), x = c.a * x1 + c.s * x0; if (i) ctx.lineTo(TX(t), XY(x)); else ctx.moveTo(TX(t), XY(x)); }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // 참 주변 궤적(RK4 200단계)
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1.6;
    for (const x0 of show) {
      let x = x0; const h = 1 / 200;
      ctx.beginPath(); ctx.moveTo(TX(0), XY(x));
      for (let k = 0; k < 200; k++) {
        const t = k * h, k1 = v(x, t), k2 = v(x + 0.5 * h * k1, t + 0.5 * h), k3 = v(x + 0.5 * h * k2, t + 0.5 * h), k4 = v(x + h * k3, t + h);
        x += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        ctx.lineTo(TX(t + h), XY(x));
      }
      ctx.stroke();
    }
    // Euler n단계 궤적
    const n = STEPS[si];
    ctx.strokeStyle = C.edge; ctx.fillStyle = C.edge; ctx.lineWidth = 2;
    for (const x0 of show) {
      let x = x0;
      const pts: [number, number][] = [[TX(0), XY(x)]];
      for (let k = 0; k < n; k++) { x += v(x, k / n) / n; pts.push([TX((k + 1) / n), XY(x)]); }
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.stroke();
      if (n <= 16) pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, 7); ctx.fill(); });
    }
    ctx.restore(); ctx.lineWidth = 1;
    // 끝점 히스토그램과 데이터 밀도
    const NB = 60, bw = (Y1 - Y0) / NB, cnt = new Array(NB).fill(0);
    for (const x of ends) { const b = Math.floor((x - Y0) / bw); if (b >= 0 && b < NB) cnt[b]++; }
    const dens = (x: number) => PI.reduce((s, p, k) => s + (p * Math.exp(-0.5 * ((x - M[k]) / S[k]) ** 2)) / (S[k] * Math.sqrt(2 * Math.PI)), 0);
    const dmax = Math.max(dens(2), ...cnt.map((c) => c / (ends.length * bw)));
    const HX = (d: number) => hx + (d / dmax) * (histW - 8);
    ctx.fillStyle = alpha(C.edge, 0.45);
    cnt.forEach((c, b) => { const d = c / (ends.length * bw); if (d > 0) ctx.fillRect(hx, XY(Y0 + (b + 1) * bw), HX(d) - hx, Math.max(1, XY(Y0 + b * bw) - XY(Y0 + (b + 1) * bw) - 0.5)); });
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const x = Y0 + ((Y1 - Y0) * i) / 200; if (i) ctx.lineTo(HX(dens(x)), XY(x)); else ctx.moveTo(HX(dens(x)), XY(x)); }
    ctx.stroke(); ctx.lineWidth = 1;
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(hx, g.y); ctx.lineTo(hx, g.y + g.h); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("끝점", hx, g.y - 6);
    let lx = g.x;
    for (const [t, col] of [["Euler", C.edge], ["참 궤적", C.mid], ...(showCond ? [["조건부 경로", C.ink2]] : [])] as [string, string][]) {
      ctx.fillStyle = col; ctx.fillText(t, lx, g.y - 6); lx += ctx.measureText(t).width + 12;
    }
  }

  function update() {
    const v = vf();
    const n = STEPS[si];
    ends = euler(v, X0S, n);
    const W = w1(ends, REF);
    ro.set("w", `W₁(${n} step) = ${W.toFixed(3)}`, W < 0.02 ? "ok" : "hot");
    ro.set("w1", `1 step ${w1(euler(v, X0S, 1), REF).toFixed(3)} · 32 step ${w1(euler(v, X0S, 32), REF).toFixed(3)}`);
    ro.set("fl", reflow ? "CFM 손실 바닥 0 (짝이 교차하지 않음)" : `CFM 손실 바닥 ${floors[path].toFixed(3)}`);
    draw();
  }

  const pp = presets<Path>([{ key: "line", label: "직선 경로" }, { key: "trig", label: "삼각 경로" }], (k) => { path = k; update(); }, path);
  pp.row.classList.add("segmented");
  const ps = param({ label: "step", min: 0, max: STEPS.length - 1, step: 1, value: si, fmt: (i) => String(STEPS[i]), aria: "Euler step 수", onInput: (i) => { si = i; update(); } });
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const tools = el("div", "wframe-row");
  const tg = (label: string, on: boolean, fn: (v: boolean) => void) => {
    const l = el("label", "toggle");
    l.innerHTML = `<input type="checkbox"${on ? " checked" : ""} /> `;
    l.append(label);
    l.querySelector("input")!.addEventListener("change", (e) => fn((e.target as HTMLInputElement).checked));
    return l;
  };
  tools.append(tg("reflow 짝", reflow, (v) => { reflow = v; update(); }), tg("조건부 경로", showCond, (v) => { showCond = v; draw(); }));
  const ro = readouts();
  f.panel.append(pp.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
