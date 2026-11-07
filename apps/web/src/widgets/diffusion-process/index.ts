// 확산 과정의 forward와 reverse (data/widgets.toml: diffusion-process).
// 1차원 가우시안 혼합 p₀에 VP 잡음 스케줄 β(t) = β₀ + (β₁ − β₀)t (β₀ = 0.1, β₁ = 20, t ∈ [0, 1])을 적용한다.
//   forward: x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε,  ᾱ_t = exp(−∫₀ᵗ β(s) ds).  입자는 격자 시각 사이의 정확한 OU 전이로 움직인다.
//   p_t도 가우시안 혼합이라 밀도와 score ∇log p_t를 닫힌 형태로 계산한다(배경 = 정확한 p_t, 열마다 최댓값으로 정규화).
//   reverse SDE(Euler–Maruyama): x ← x + h[½β x + β ∇log p_t(x)] + √(βh) z
//   reverse ODE(probability flow, Euler): x ← x + h[½β x + ½β ∇log p_t(x)]
// 학습된 신경망 대신 정확한 score를 쓰므로, 남는 오차는 step 수 N(이산화)과 유한한 입자 수뿐이다.
import { frame, param, presets, readouts, el, C, rng, rgbOf, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

const PI = [0.35, 0.65], MU = [-1.5, 1.2], SD = [0.25, 0.35];
const B0 = 0.1, B1 = 20;
const M = 2000, KF = 200, SHOW = 48;
const XR = 3.6;
type Mode = "forward" | "sde" | "ode";

const abar = (t: number) => Math.exp(-(B0 * t + 0.5 * (B1 - B0) * t * t));
const beta = (t: number) => B0 + (B1 - B0) * t;
/** p_t의 성분: [가중치, 평균, 분산] */
const comps = (t: number) => { const a = abar(t); return PI.map((w, k) => [w, Math.sqrt(a) * MU[k], a * SD[k] ** 2 + 1 - a]); };
const density = (x: number, cs: number[][]) => cs.reduce((s, [w, m, v]) => s + (w * Math.exp((-0.5 * (x - m) ** 2) / v)) / Math.sqrt(2 * Math.PI * v), 0);
function score(x: number, cs: number[][]) {
  let num = 0, den = 0;
  for (const [w, m, v] of cs) { const p = (w * Math.exp((-0.5 * (x - m) ** 2) / v)) / Math.sqrt(v); num -= (p * (x - m)) / v; den += p; }
  return den > 1e-300 ? num / den : -x;
}
// erf: Abramowitz–Stegun 7.1.26 (최대 오차 1.5e-7)
function erf(x: number) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}
const cdf = (x: number, cs: number[][]) => cs.reduce((s, [w, m, v]) => s + w * 0.5 * (1 + erf((x - m) / Math.sqrt(2 * v))), 0);
function quantile(u: number, cs: number[][]) {
  let lo = -8, hi = 8;
  for (let i = 0; i < 50; i++) { const mid = 0.5 * (lo + hi); if (cdf(mid, cs) < u) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

/** forward 경로: KF+1개 시각(0, 1/KF, …, 1)에서 입자 M개의 위치 */
function forwardPaths(): Float64Array[] {
  const r = rng(777);
  let x = new Float64Array(M);
  for (let m = 0; m < M; m++) { const k = r.u() < PI[0] ? 0 : 1; x[m] = MU[k] + SD[k] * r.normal(); }
  const out = [x];
  for (let j = 1; j <= KF; j++) {
    const ratio = abar(j / KF) / abar((j - 1) / KF), a = Math.sqrt(ratio), b = Math.sqrt(1 - ratio);
    const y = new Float64Array(M);
    for (let m = 0; m < M; m++) y[m] = a * x[m] + b * r.normal();
    out.push(y); x = y;
  }
  return out;
}
/** reverse 경로: t = 1에서 0까지 N step. out[j]는 시각 1 − j/N의 위치 */
function reversePaths(n: number, sde: boolean): Float64Array[] {
  const r = rng(9007);
  let x = new Float64Array(M);
  for (let m = 0; m < M; m++) x[m] = r.normal();
  const out = [x], h = 1 / n;
  for (let j = 0; j < n; j++) {
    const t = 1 - j * h, bt = beta(t), cs = comps(t), y = new Float64Array(M);
    for (let m = 0; m < M; m++) {
      const s = score(x[m], cs);
      y[m] = sde ? x[m] + h * (0.5 * bt * x[m] + bt * s) + Math.sqrt(bt * h) * r.normal() : x[m] + h * (0.5 * bt * x[m] + 0.5 * bt * s);
    }
    out.push(y); x = y;
  }
  return out;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "확산 과정의 forward와 reverse", "simulate", narrow ? 0.95 : 0.6, "시간 t에 따른 정확한 주변분포 p_t의 밀도와 입자 경로, 현재 시각의 히스토그램");
  let mode: Mode = "forward", t = 0, logN = 6;
  const fwd = forwardPaths();
  let paths = fwd;
  const N = () => Math.round(2 ** logN);
  /** 현재 모드에서 시각 t의 입자 위치(격자 사이는 선형 보간) */
  function at(tt: number): Float64Array {
    const n = paths.length - 1;
    const g = mode === "forward" ? tt * n : (1 - tt) * n;
    const j = Math.min(n - 1, Math.floor(g)), w = g - j;
    if (w <= 1e-12) return paths[j];
    const a = paths[j], b = paths[j + 1], out = new Float64Array(M);
    for (let m = 0; m < M; m++) out[m] = a[m] * (1 - w) + b[m] * w;
    return out;
  }

  // 배경: 정확한 p_t(x), 열마다 최댓값으로 나눈 값
  const GT = 160, GX = 120;
  const heat = new Float32Array(GT * GX);
  for (let i = 0; i < GT; i++) {
    const cs = comps((i + 0.5) / GT);
    let mx = 0;
    for (let j = 0; j < GX; j++) { const v = density(XR - ((j + 0.5) / GX) * 2 * XR, cs); heat[j * GT + i] = v; mx = Math.max(mx, v); }
    for (let j = 0; j < GX; j++) heat[j * GT + i] /= mx;
  }
  const off = document.createElement("canvas");
  off.width = GT; off.height = GX;
  const octx = off.getContext("2d")!;
  function paintHeat() {
    const img = octx.createImageData(GT, GX), [r, g, b] = rgbOf(C.ink2);
    for (let k = 0; k < GT * GX; k++) { img.data[4 * k] = r; img.data[4 * k + 1] = g; img.data[4 * k + 2] = b; img.data[4 * k + 3] = Math.round(255 * 0.75 * Math.pow(heat[k], 0.8)); }
    octx.putImageData(img, 0, 0);
  }

  let cur = at(0);
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    paintHeat();
    const pad = 12, top = 26, bot = 26, gap = 10;
    const sideW = Math.round(Math.max(64, W * 0.22));
    const A = { x: pad + 18, y: top, w: W - 2 * pad - 18 - sideW - gap, h: H - top - bot };
    const X = (tt: number) => A.x + tt * A.w;
    const Y = (x: number) => A.y + ((XR - x) / (2 * XR)) * A.h;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, A.x, A.y, A.w, A.h);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(A.x + 0.5, A.y + 0.5, A.w - 1, A.h - 1);
    // 입자 경로: 지나온 구간만
    const n = paths.length - 1;
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
    const stride = Math.floor(M / SHOW);
    for (let s = 0; s < SHOW; s++) {
      const m = s * stride;
      ctx.beginPath();
      for (let j = 0; j <= n; j++) {
        const tj = mode === "forward" ? j / n : 1 - j / n;
        if (mode === "forward" ? tj > t + 1e-9 : tj < t - 1e-9) break;
        const px = X(tj), py = Y(paths[j][m]);
        if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.lineTo(X(t), Y(cur[m]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // 현재 시각
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(t), A.y); ctx.lineTo(X(t), A.y + A.h); ctx.stroke();
    ctx.fillStyle = C.edge;
    for (let s = 0; s < SHOW; s++) { ctx.beginPath(); ctx.arc(X(t), Y(cur[s * stride]), 2.4, 0, 7); ctx.fill(); }
    // 축
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("t = 0", A.x + 16, A.y + A.h + 16);
    ctx.fillText("1", A.x + A.w - 4, A.y + A.h + 16);
    ctx.textAlign = "right";
    for (const v of [-3, 0, 3]) ctx.fillText(String(v), A.x - 5, Y(v) + 4);
    ctx.textAlign = "left";
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText(mode === "forward" ? "x_t: 데이터 → 잡음" : "x_t: 잡음 → 데이터", A.x, A.y - 9);

    // 옆: 시각 t의 히스토그램(입자)과 정확한 p_t
    const B = { x: A.x + A.w + gap, y: A.y, w: sideW, h: A.h };
    const bins = 48, cnt = new Float64Array(bins), bw = (2 * XR) / bins;
    for (let m = 0; m < M; m++) { const k = Math.floor((XR - cur[m]) / bw); if (k >= 0 && k < bins) cnt[k]++; }
    const cs = comps(t);
    let pmax = 0;
    for (let j = 0; j <= 100; j++) pmax = Math.max(pmax, density(-XR + (2 * XR * j) / 100, cs));
    const hmax = Math.max(pmax, ...Array.from(cnt, (c) => c / (M * bw))) * 1.05;
    ctx.fillStyle = C.edgeSoft;
    for (let k = 0; k < bins; k++) {
      const d = cnt[k] / (M * bw);
      ctx.fillRect(B.x, B.y + (k / bins) * B.h + 0.5, (d / hmax) * B.w, B.h / bins - 1);
    }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let j = 0; j <= 120; j++) {
      const x = XR - (2 * XR * j) / 120, px = B.x + (density(x, cs) / hmax) * B.w, py = Y(x);
      if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(B.x + 0.5, B.y); ctx.lineTo(B.x + 0.5, B.y + B.h); ctx.stroke();
    ctx.fillStyle = C.mid;
    ctx.fillText("p_t", B.x, B.y - 9);
  }

  function update() {
    cur = at(t);
    const a = abar(t), cs = comps(t);
    // 입자의 경험분포와 정확한 p_t 사이의 1-Wasserstein 거리(분위수 짝짓기)
    const xs = Float64Array.from(cur).sort();
    let w1 = 0;
    for (let i = 0; i < M; i++) w1 += Math.abs(xs[i] - quantile((i + 0.5) / M, cs));
    w1 /= M;
    let left = 0;
    for (let m = 0; m < M; m++) if (cur[m] < 0) left++;
    ro.set("t", `t ${t.toFixed(2)} · ᾱ_t ${a < 1e-3 ? a.toExponential(1) : a.toFixed(3)}`, "hot");
    ro.set("coef", `√ᾱ ${Math.sqrt(a).toFixed(2)} · √(1−ᾱ) ${Math.sqrt(1 - a).toFixed(2)}`);
    ro.set("left", `x < 0 비율 ${Math.round((left / M) * 100)}% (p_t ${Math.round(cdf(0, cs) * 100)}%)`);
    ro.set("w1", `W₁(입자, p_t) ${w1.toFixed(3)}`);
    pageState.set("diffusion", { mode, t, steps: mode === "forward" ? KF : N(), w1 }, root);
    draw();
  }

  function setMode(k: Mode) {
    mode = k;
    paths = k === "forward" ? fwd : reversePaths(N(), k === "sde");
    nRow.hidden = k === "forward";
  }

  let anim = 0;
  function playSweep() {
    cancelAnimationFrame(anim);
    const [from, to] = mode === "forward" ? [0, 1] : [1, 0];
    if (reducedMotion()) { t = to; pt.set(t); update(); return; }
    const t0 = performance.now();
    const step = (now: number) => {
      if (!f.canvas.isConnected) return;
      const k = Math.min(1, (now - t0) / 2400);
      t = Math.round((from + (to - from) * k) * 200) / 200;
      pt.set(t); update();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  const modeBtns = presets<Mode>([{ key: "forward", label: "forward" }, { key: "sde", label: narrow ? "SDE" : "reverse SDE" }, { key: "ode", label: narrow ? "ODE" : "reverse ODE" }], (k) => {
    setMode(k);
    t = k === "forward" ? 0 : 1; pt.set(t); update();
  }, mode);
  modeBtns.row.classList.add("segmented");
  const pt = param({ label: "t", min: 0, max: 1, step: 0.005, value: t, aria: "확산 시각 t", onInput: (v) => { cancelAnimationFrame(anim); t = v; update(); } });
  const pn = param({ label: "N", min: 2, max: 9, step: 0.5, value: logN, fmt: (v) => String(Math.round(2 ** v)), aria: "reverse 샘플링의 step 수 N", onInput: (v) => { logN = v; setMode(mode); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const nRow = el("div", "param-grid");
  nRow.append(pn.row);
  nRow.hidden = true;
  const tools = el("div", "wframe-row");
  const play = el("button", "preset");
  play.type = "button";
  play.textContent = "▶ 재생";
  play.addEventListener("click", playSweep);
  tools.append(play);
  const ro = readouts();
  f.panel.append(modeBtns.row, grid, nRow, tools, ro.row);
  f.onResize(draw);
  update();
}
