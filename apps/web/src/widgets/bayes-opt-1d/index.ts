// 1D 베이지안 최적화 (data/widgets.toml: bayes-opt-1d).
// 10.7장 셀과 같은 설정: 목적함수 f(x) = sin 3x + 0.5 sin 10x + 0.3x를 [0, 2]에서 최소화, 후보 401개,
// RBF 커널(ℓ = 0.15, 사전 분산 1) GP, y를 표준화, 잡음 1e−6. 획득함수는 EI(ξ) 또는 LCB = μ − κσ.
// 위: 참 함수(흐린 선), GP 사후 평균과 ±2σ, 관측점, 다음 평가점. 아래: 획득함수. 조작: 다음 평가 버튼, 획득함수 스위치, ξ·κ 슬라이더.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

export const sim = (x: number) => Math.sin(3 * x) + 0.5 * Math.sin(10 * x) + 0.3 * x;
const NC = 401;
export const CAND = Array.from({ length: NC }, (_, i) => (2 * i) / (NC - 1));
const FSTAR = Math.min(...CAND.map(sim));
const ELL = 0.15, BUDGET = 15;
export const X_INIT = [0.25, 1.0, 1.45];
const kern = (a: number, b: number) => Math.exp((-0.5 * (a - b) ** 2) / (ELL * ELL));

/** GP 사후 평균·표준편차(원래 단위). 10.7장 gp_posterior와 같다. */
export function gpPosterior(X: number[], y: number[], noise = 1e-6) {
  const n = X.length;
  const m = y.reduce((s, v) => s + v, 0) / n;
  const s = Math.sqrt(y.reduce((t, v) => t + (v - m) ** 2, 0) / n) + 1e-12;
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = kern(X[i], X[j]) + (i === j ? noise : 0);
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = K[i * n + j];
    for (let k = 0; k < j; k++) v -= L[i * n + k] * L[j * n + k];
    L[i * n + j] = i === j ? Math.sqrt(Math.max(v, 1e-14)) : v / L[j * n + j];
  }
  const fwd = (b: ArrayLike<number>) => { const x = new Float64Array(n); for (let i = 0; i < n; i++) { let v = b[i]; for (let k = 0; k < i; k++) v -= L[i * n + k] * x[k]; x[i] = v / L[i * n + i]; } return x; };
  const alphaV = fwd(y.map((v) => (v - m) / s));
  const mu = new Float64Array(NC), sd = new Float64Array(NC);
  for (let t = 0; t < NC; t++) {
    const v = fwd(X.map((xi) => kern(xi, CAND[t])));
    let mm = 0, vv = 0;
    for (let i = 0; i < n; i++) { mm += v[i] * alphaV[i]; vv += v[i] * v[i]; }
    mu[t] = m + s * mm;
    sd[t] = s * Math.sqrt(Math.max(1 - vv, 1e-12));
  }
  return { mu, sd };
}
const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
/** 표준정규 누적분포(erfc 근사, 상대오차 1e−7 수준: Numerical Recipes erfcc) */
export function Phi(z: number) {
  const x = -z / Math.SQRT2, t = 1 / (1 + 0.5 * Math.abs(x));
  const r = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return 0.5 * (x >= 0 ? r : 2 - r);
}
export type Acq = "ei" | "lcb";
/** 획득 점수(클수록 다음 평가점): EI 그대로, LCB는 부호를 바꾼 −(μ − κσ) */
export function acquisition(acq: Acq, mu: Float64Array, sd: Float64Array, best: number, xi: number, kappa: number) {
  const a = new Float64Array(NC);
  for (let t = 0; t < NC; t++) {
    if (acq === "ei") { const z = (best - mu[t] - xi) / sd[t]; a[t] = (best - mu[t] - xi) * Phi(z) + sd[t] * phi(z); }
    else a[t] = -(mu[t] - kappa * sd[t]);
  }
  return a;
}
export function nextPoint(X: number[], acq: Acq, xi: number, kappa: number) {
  const y = X.map(sim);
  const { mu, sd } = gpPosterior(X, y);
  const a = acquisition(acq, mu, sd, Math.min(...y), xi, kappa);
  let k = 0;
  for (let t = 1; t < NC; t++) if (a[t] > a[k]) k = t;
  return { mu, sd, a, k };
}

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "1D 베이지안 최적화", "stepper", narrow ? 0.82 : 0.62, "참 목적함수, GP 사후 평균과 ±2σ 띠, 관측점, 그리고 획득함수와 다음 평가점");
  let X = [...X_INIT], acq: Acq = "ei", xi = 0.01, kappa = 2;
  let st = nextPoint(X, acq, xi, kappa);
  const Y0 = -2.2, Y1 = 2.2;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 34, R = 12, T = 12, gap = 24, h1 = Math.round((H - T - 22 - gap) * 0.64);
    const top = { y: T, h: h1 }, bot = { y: T + h1 + gap, h: H - T - h1 - gap - 22 };
    const X_ = (x: number) => L + (x / 2) * (W - L - R);
    const Y = (y: number) => top.y + ((Y1 - y) / (Y1 - Y0)) * top.h;
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    ctx.strokeRect(L, top.y, W - L - R, top.h); ctx.strokeRect(L, bot.y, W - L - R, bot.h);
    ctx.textAlign = "right";
    for (const v of [-2, -1, 0, 1, 2]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillText(String(v).replace("-", "−"), L - 5, Y(v) + 4); }
    ctx.textAlign = "center";
    for (const v of [0, 0.5, 1, 1.5, 2]) ctx.fillText(String(v), X_(v), H - 6);
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(L, top.y, W - L - R, top.h); ctx.clip();
    // ±2σ 띠
    ctx.fillStyle = alpha(C.edge, 0.16);
    ctx.beginPath();
    CAND.forEach((x, t) => (t ? ctx.lineTo(X_(x), Y(st.mu[t] + 2 * st.sd[t])) : ctx.moveTo(X_(x), Y(st.mu[t] + 2 * st.sd[t]))));
    for (let t = NC - 1; t >= 0; t--) ctx.lineTo(X_(CAND[t]), Y(st.mu[t] - 2 * st.sd[t]));
    ctx.closePath(); ctx.fill();
    // 참 함수
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
    ctx.beginPath(); CAND.forEach((x, t) => (t ? ctx.lineTo(X_(x), Y(sim(x))) : ctx.moveTo(X_(x), Y(sim(x))))); ctx.stroke(); ctx.setLineDash([]);
    // 사후 평균
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); CAND.forEach((x, t) => (t ? ctx.lineTo(X_(x), Y(st.mu[t])) : ctx.moveTo(X_(x), Y(st.mu[t])))); ctx.stroke();
    // 다음 평가점
    const xn = CAND[st.k];
    if (X.length < BUDGET) {
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X_(xn), top.y); ctx.lineTo(X_(xn), top.y + top.h); ctx.stroke();
    }
    // 관측점(마지막 것은 고리로)
    const ys = X.map(sim);
    const ib = ys.indexOf(Math.min(...ys));
    X.forEach((x, i) => {
      ctx.fillStyle = i === ib ? C.good : C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X_(x), Y(ys[i]), 4.5, 0, 7); ctx.fill(); ctx.stroke();
      if (i === X.length - 1 && X.length > X_INIT.length) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(X_(x), Y(ys[i]), 8, 0, 7); ctx.stroke(); }
    });
    ctx.restore();
    // 범례: 참 함수(점선) · 사후 평균 μ · ±2σ 띠
    {
      let lx = L + 8;
      const ly = top.y + 14;
      ctx.lineWidth = 1.3; ctx.strokeStyle = C.mid; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(lx, ly - 4); ctx.lineTo(lx + 18, ly - 4); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.ink; ctx.fillText("f(x)", lx + 22, ly); lx += 22 + ctx.measureText("f(x)").width + 14;
      ctx.lineWidth = 2.4; ctx.strokeStyle = C.edge;
      ctx.beginPath(); ctx.moveTo(lx, ly - 4); ctx.lineTo(lx + 18, ly - 4); ctx.stroke();
      ctx.fillText("μ(x)", lx + 22, ly); lx += 22 + ctx.measureText("μ(x)").width + 14;
      ctx.fillStyle = alpha(C.edge, 0.25); ctx.fillRect(lx, ly - 9, 18, 10);
      ctx.fillStyle = C.ink; ctx.fillText("±2σ", lx + 22, ly);
    }
    // 아래: 획득함수
    let lo = Infinity, hi = -Infinity;
    const show = Array.from(st.a, (v) => (acq === "ei" ? v : -v)); // LCB는 μ − κσ 그대로 그린다
    show.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
    if (acq === "ei") lo = 0;
    if (hi - lo < 1e-9) hi = lo + 1e-9;
    const YA = (v: number) => bot.y + 4 + ((hi - v) / (hi - lo)) * (bot.h - 8);
    ctx.save(); ctx.beginPath(); ctx.rect(L, bot.y, W - L - R, bot.h); ctx.clip();
    ctx.fillStyle = alpha(C.ink2, 0.22); ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath();
    CAND.forEach((x, t) => (t ? ctx.lineTo(X_(x), YA(show[t])) : ctx.moveTo(X_(x), YA(show[t]))));
    ctx.stroke();
    if (acq === "ei") { ctx.lineTo(X_(2), YA(0)); ctx.lineTo(X_(0), YA(0)); ctx.closePath(); ctx.fill(); }
    if (X.length < BUDGET) {
      ctx.fillStyle = C.ink2; ctx.strokeStyle = C.bg;
      ctx.beginPath(); ctx.arc(X_(xn), YA(show[st.k]), 5, 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    ctx.fillStyle = C.mid;
    ctx.fillText(acq === "ei" ? `EI(x), ξ = ${xi.toFixed(2)} · 최대점` : `LCB(x) = μ − ${kappa.toFixed(1)}σ · 최소점`, L + 6, bot.y + 14);
  }

  function update() {
    st = nextPoint(X, acq, xi, kappa);
    const ys = X.map(sim);
    const ib = ys.indexOf(Math.min(...ys));
    const regret = ys[ib] - FSTAR;
    ro.set("n", `평가 ${X.length} / ${BUDGET}번`);
    ro.set("best", `최솟값 f⁺ = ${ys[ib].toFixed(3).replace("-", "−")} (x = ${X[ib].toFixed(3)})`);
    ro.set("regret", `regret ${regret < 5e-4 ? regret.toExponential(1).replace("e-", "e−") : regret.toFixed(3)}`, regret < 0.01 ? "ok" : "");
    ro.set("next", X.length < BUDGET ? `다음 x = ${CAND[st.k].toFixed(3)}` : "예산을 다 썼다", X.length < BUDGET ? "hot" : "");
    nextBtn.disabled = five.disabled = X.length >= BUDGET;
    pxi.row.hidden = acq !== "ei";
    pk.row.hidden = acq !== "lcb";
    pageState.set("bayesOpt", { X: [...X], acq, xi, kappa }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;
  const evalOnce = () => { if (X.length < BUDGET) { X.push(CAND[st.k]); st = nextPoint(X, acq, xi, kappa); } };

  const ab = presets<Acq>([{ key: "ei", label: "EI" }, { key: "lcb", label: "LCB (μ − κσ)" }], (k) => { acq = k; update(); }, acq);
  ab.row.classList.add("segmented");
  const pxi = param({ label: "ξ", min: 0, max: 0.5, step: 0.01, value: xi, aria: "EI의 ξ(개선 여유)", onInput: (v) => { xi = v; update(); } });
  const pk = param({ label: "κ", min: 0, max: 5, step: 0.1, value: kappa, fmt: (v) => v.toFixed(1), aria: "LCB의 κ", onInput: (v) => { kappa = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pxi.row, pk.row);
  const tools = el("div", "wframe-row");
  const nextBtn = el("button", "preset");
  nextBtn.type = "button"; nextBtn.textContent = "다음 평가 →";
  nextBtn.addEventListener("click", () => { evalOnce(); update(); });
  const five = el("button", "preset");
  five.type = "button"; five.textContent = "5번 평가";
  five.addEventListener("click", () => { for (let i = 0; i < 5; i++) evalOnce(); update(); });
  tools.append(nextBtn, five);
  const ro = readouts();
  f.panel.append(ab.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
