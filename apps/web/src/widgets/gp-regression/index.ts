// Gaussian process 회귀 (data/widgets.toml: gp-regression).
// 사전 f ~ GP(0, k), k(x, x') = σ_f² exp(−(x − x')² / (2ℓ²)) (σ_f = 1), 관측 y = f(x) + ε, ε ~ N(0, σ_n²).
// 사후 평균 μ(x*) = k*ᵀ (K + σ_n² I)⁻¹ y, 분산 σ²(x*) = k(x*, x*) − k*ᵀ (K + σ_n² I)⁻¹ k*를 Cholesky 분해로 계산한다.
// 띠는 잠재함수 f의 μ ± 2σ(관측 잡음은 포함하지 않는다). 그림을 누르면 관측점을 더하고, 있는 점을 누르면 지운다.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

const XR = 5, YR = 3, M = 161, MAXN = 40;
const GRID = Array.from({ length: M }, (_, i) => -XR + (2 * XR * i) / (M - 1));
const rbf = (a: number, b: number, ell: number) => Math.exp(-((a - b) ** 2) / (2 * ell * ell));

/** 대칭 양의 정부호 행렬(행 우선, n×n)의 하삼각 Cholesky 인자 */
function cholesky(A: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-12));
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  return L;
}
/** L x = b (전진 대입) */
function solveL(L: Float64Array, n: number, b: ArrayLike<number>) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}
/** Lᵀ x = b (후진 대입) */
function solveLT(L: Float64Array, n: number, b: ArrayLike<number>) {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

/** GP 사후: 격자 위 평균·분산, 로그 주변우도, (요청하면) 사후 공분산 */
export function gpPosterior(xs: number[], ys: number[], ell: number, sn: number, grid: number[] = GRID, wantCov = false) {
  const n = xs.length, m = grid.length;
  const mean = new Float64Array(m), variance = new Float64Array(m).fill(1);
  let V = new Float64Array(0), lml = 0;
  if (n) {
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = rbf(xs[i], xs[j], ell) + (i === j ? sn * sn : 0);
    const L = cholesky(K, n);
    const a = solveLT(L, n, solveL(L, n, ys));
    V = new Float64Array(n * m);
    for (let t = 0; t < m; t++) {
      const ks = xs.map((x) => rbf(x, grid[t], ell));
      mean[t] = ks.reduce((s, k, i) => s + k * a[i], 0);
      const v = solveL(L, n, ks);
      let vv = 0;
      for (let i = 0; i < n; i++) { V[i * m + t] = v[i]; vv += v[i] * v[i]; }
      variance[t] = Math.max(0, 1 - vv);
    }
    let logdet = 0;
    for (let i = 0; i < n; i++) logdet += Math.log(L[i * n + i]);
    lml = -0.5 * ys.reduce((s, y, i) => s + y * a[i], 0) - logdet - 0.5 * n * Math.log(2 * Math.PI);
  }
  let cov: Float64Array | null = null;
  if (wantCov) {
    cov = new Float64Array(m * m);
    for (let s = 0; s < m; s++) for (let t = 0; t <= s; t++) {
      let c = rbf(grid[s], grid[t], ell);
      for (let i = 0; i < n; i++) c -= V[i * m + s] * V[i * m + t];
      cov[s * m + t] = cov[t * m + s] = c;
    }
  }
  return { mean, variance, lml, cov };
}

type Pt = [number, number];
const sinPts = (xs: number[]): Pt[] => xs.map((x, i) => [x, Math.round((Math.sin(1.1 * x) + 0.15 * Math.cos(7 * i)) * 100) / 100]);
const DATA = {
  three: { label: "점 3개", pts: [[-3, -0.8], [-0.5, 1], [2, 0.4]] as Pt[] },
  sine: { label: "사인 8점", pts: sinPts([-4.2, -3, -2.1, -1.2, 0, 1.1, 2.4, 3.6]) },
  gap: { label: "가운데 빈 구간", pts: sinPts([-4.5, -3.8, -3.1, -2.5, 2.5, 3.2, 3.9, 4.6]) },
  none: { label: "관측 없음", pts: [] as Pt[] },
} as const;
type DK = keyof typeof DATA;

export function mount(root: HTMLElement) {
  const f = frame(root, "Gaussian process 회귀", "manipulate", 0.6, "GP 사후 평균과 ±2σ 띠, 관측점, 사후 표본 함수");
  let pts: Pt[] = DATA.sine.pts.map((p) => [...p] as Pt);
  let ell = 1, sn = 0.1, showSamples = false;
  const r = rng(99);
  const Z = Array.from({ length: 3 }, () => Array.from({ length: M }, () => r.normal()));
  let post = gpPosterior([], [], ell, sn);
  let samples: Float64Array[] = [];

  const PAD_L = 30, PAD_R = 12, PAD_T = 12, PAD_B = 24;
  const X = (v: number) => PAD_L + ((v + XR) / (2 * XR)) * (f.width - PAD_L - PAD_R);
  const Y = (v: number) => PAD_T + ((YR - v) / (2 * YR)) * (f.height - PAD_T - PAD_B);
  const toW = (px: number, py: number): Pt => [-XR + ((px - PAD_L) / (f.width - PAD_L - PAD_R)) * 2 * XR, YR - ((py - PAD_T) / (f.height - PAD_T - PAD_B)) * 2 * YR];

  function compute() {
    post = gpPosterior(pts.map((p) => p[0]), pts.map((p) => p[1]), ell, sn, GRID, showSamples);
    samples = [];
    if (showSamples && post.cov) {
      const cov = post.cov;
      for (let i = 0; i < M; i++) cov[i * M + i] += 1e-6; // 수치 안정용 jitter
      const L = cholesky(cov, M);
      samples = Z.map((z) => {
        const s = new Float64Array(M);
        for (let i = 0; i < M; i++) { let v = post.mean[i]; for (let k = 0; k <= i; k++) v += L[i * M + k] * z[k]; s[i] = v; }
        return s;
      });
    }
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (let k = -XR; k <= XR; k++) {
      ctx.beginPath(); ctx.moveTo(X(k), PAD_T); ctx.lineTo(X(k), H - PAD_B); ctx.stroke();
      if (k % (W < 480 ? 2 : 1) === 0) { const s = String(k); ctx.fillText(s, X(k) - ctx.measureText(s).width / 2, H - PAD_B + 14); }
    }
    for (let k = -YR; k <= YR; k++) {
      ctx.beginPath(); ctx.moveTo(PAD_L, Y(k)); ctx.lineTo(W - PAD_R, Y(k)); ctx.stroke();
      ctx.fillText(String(k), k < 0 ? 4 : 10, Y(k) + 4);
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(PAD_L, PAD_T, W - PAD_L - PAD_R, H - PAD_T - PAD_B); ctx.clip();
    // ±2σ 띠
    ctx.fillStyle = alpha(C.edge, 0.16);
    ctx.beginPath();
    GRID.forEach((g, i) => { const y = Y(post.mean[i] + 2 * Math.sqrt(post.variance[i])); i ? ctx.lineTo(X(g), y) : ctx.moveTo(X(g), y); });
    for (let i = M - 1; i >= 0; i--) ctx.lineTo(X(GRID[i]), Y(post.mean[i] - 2 * Math.sqrt(post.variance[i])));
    ctx.closePath(); ctx.fill();
    // 사전 평균 0
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, Y(0)); ctx.lineTo(W - PAD_R, Y(0)); ctx.stroke(); ctx.setLineDash([]);
    // 사후 표본
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.3;
    for (const s of samples) { ctx.beginPath(); GRID.forEach((g, i) => (i ? ctx.lineTo(X(g), Y(s[i])) : ctx.moveTo(X(g), Y(s[i])))); ctx.stroke(); }
    // 사후 평균
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6;
    ctx.beginPath(); GRID.forEach((g, i) => (i ? ctx.lineTo(X(g), Y(post.mean[i])) : ctx.moveTo(X(g), Y(post.mean[i])))); ctx.stroke();
    // 관측점
    for (const [px, py] of pts) {
      ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(px), Y(py), 4.5, 0, 7); ctx.fill();
      ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.restore();
  }

  function update() {
    compute();
    let im = 0;
    post.variance.forEach((v, i) => { if (v > post.variance[im]) im = i; });
    ro.set("n", `관측 ${pts.length}개`);
    ro.set("lml", pts.length ? `log p(y | X) = ${post.lml.toFixed(2)}` : "log p(y | X) 없음", "hot");
    ro.set("sd", `최대 σ(x) = ${Math.sqrt(post.variance[im]).toFixed(2)} (x = ${GRID[im].toFixed(2)})`);
    pageState.set("gp", { lengthscale: ell, noise: sn, points: pts.map((p) => [...p]) }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const pre = presets((Object.keys(DATA) as DK[]).map((k) => ({ key: k, label: DATA[k].label })), (k) => {
    pts = DATA[k].pts.map((p) => [...p] as Pt); update();
  }, "sine");
  const pl = param({ label: "ℓ", min: 0.2, max: 3, step: 0.05, value: ell, aria: "길이척도 ℓ", onInput: (v) => { ell = v; update(); } });
  const ps = param({ label: "σₙ", min: 0.01, max: 1, step: 0.01, value: sn, aria: "관측 잡음 표준편차 σₙ", onInput: (v) => { sn = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pl.row, ps.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 사후 표본 3개`;
  sw.querySelector("input")!.addEventListener("change", (e) => { showSamples = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);

  f.canvas.addEventListener("pointerdown", (e) => {
    const rect = f.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const hit = pts.findIndex(([a, b]) => Math.hypot(X(a) - px, Y(b) - py) < 12);
    if (hit >= 0) pts.splice(hit, 1);
    else {
      const [wx, wy] = toW(px, py);
      if (Math.abs(wx) > XR || Math.abs(wy) > YR || pts.length >= MAXN) return;
      pts.push([Math.round(wx * 20) / 20, Math.round(wy * 20) / 20]);
    }
    pre.select(null);
    update();
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const rect = f.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    f.canvas.style.cursor = pts.some(([a, b]) => Math.hypot(X(a) - px, Y(b) - py) < 12) ? "pointer" : "crosshair";
  });
  f.onResize(draw);
  update();
  draw();
}
