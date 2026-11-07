// 반복법 수렴 비교 (data/widgets.toml: iterative-solver-race).
// 9.3장의 1차원 Poisson 행렬(대각 2, 이웃 −1)에 대각 이동 σ ≥ 0을 더한 A = (2 + σ)I − (이웃) 으로 Ax = 1을 푼다.
// σ = 0이면 본문의 Poisson 행렬 그대로이고, σ를 키우면 조건수 κ = (σ + 2 + 2cos(π/(n+1)))/(σ + 2 − 2cos(π/(n+1)))가 줄어든다.
// 조작한 κ에 맞는 σ를 구하고(κ가 Poisson의 κ보다 크면 σ = 0), Jacobi·Gauss–Seidel·CG를 x₀ = 0에서 돌려 상대 잔차를 그린다.
// 정적 반복법은 본문 코드와 같이 x_{k+1} = x_k + M⁻¹(b − A x_k)이고, 상대 잔차 10⁻⁸에서 멈춘다.
import { frame, param, readouts, el, C } from "../frame";

export const TOL = 1e-8, KMAX = 10000;
export function sigmaFor(n: number, kappa: number) {
  const c = Math.cos(Math.PI / (n + 1)), lo = 2 - 2 * c, hi = 2 + 2 * c;
  const kmax = hi / lo;
  if (kappa >= kmax) return { sigma: 0, kappa: kmax, kmax };
  const s = (hi - kappa * lo) / (kappa - 1);
  return { sigma: s, kappa: (s + hi) / (s + lo), kmax };
}

/** 세 방법의 상대 잔차 이력. res[k] = ‖b − A x_k‖/‖b‖ (k = 0, 1, …, 멈춘 반복까지) */
export function race(n: number, sigma: number) {
  const d = 2 + sigma;
  const b = new Float64Array(n).fill(1), nb = Math.sqrt(n);
  const resid = (x: Float64Array, r: Float64Array) => {
    let s = 0;
    // (A x)_i = d x_i − x_{i−1} − x_{i+1}
    for (let i = 0; i < n; i++) { r[i] = b[i] - d * x[i] + (i > 0 ? x[i - 1] : 0) + (i < n - 1 ? x[i + 1] : 0); s += r[i] * r[i]; }
    return Math.sqrt(s) / nb;
  };
  const stationary = (gs: boolean) => {
    const x = new Float64Array(n), r = new Float64Array(n), hist: number[] = [];
    for (let k = 0; k <= KMAX; k++) {
      const rel = resid(x, r);
      hist.push(rel);
      if (rel <= TOL || k === KMAX || !Number.isFinite(rel)) break;
      if (!gs) for (let i = 0; i < n; i++) x[i] += r[i] / d;               // M = diag(A)
      else {                                                              // M = tril(A): 전진 대입으로 M z = r
        let prev = 0;
        for (let i = 0; i < n; i++) { const z = (r[i] + prev) / d; x[i] += z; prev = z; }
      }
    }
    return hist;
  };
  const cg = () => {
    const x = new Float64Array(n), r = new Float64Array(b), p = new Float64Array(b), Ap = new Float64Array(n), hist = [1];
    let rr = n;
    for (let k = 0; k < KMAX; k++) {
      for (let i = 0; i < n; i++) Ap[i] = d * p[i] - (i > 0 ? p[i - 1] : 0) - (i < n - 1 ? p[i + 1] : 0);
      let pAp = 0;
      for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
      const a = rr / pAp;
      let rr2 = 0;
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; rr2 += r[i] * r[i]; }
      hist.push(Math.sqrt(rr2) / nb);
      if (Math.sqrt(rr2) / nb <= TOL) break;
      const beta = rr2 / rr;
      for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
      rr = rr2;
    }
    return hist;
  };
  return { jacobi: stationary(false), gs: stationary(true), cg: cg() };
}

const kfmt = (v: number) => (v >= 1000 ? v.toExponential(1).replace("e+", "e") : v.toFixed(1));

const METHODS = [
  { key: "jacobi", label: "Jacobi" },
  { key: "gs", label: "Gauss–Seidel" },
  { key: "cg", label: "CG" },
] as const;
type MK = (typeof METHODS)[number]["key"];

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 480;
  const f = frame(root, "반복법 수렴 비교", "compare", narrow ? 0.95 : 0.56, "Jacobi, Gauss–Seidel, CG의 상대 잔차를 반복 수에 따라 로그–로그 눈금으로 그린 곡선");
  let n = Number(attrs.n ?? 50), lk = 4;
  let res = race(n, 0);
  let info = sigmaFor(n, 10 ** lk);
  let pending = false;
  const col = (k: MK) => (k === "jacobi" ? C.mid : k === "gs" ? C.ink2 : C.edge);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const x0 = 46, x1 = W - 14, y0 = 30, y1 = H - 30;
    let peak = 1;
    for (const m of METHODS) for (const v of res[m.key]) if (v > peak) peak = v;
    const LX = Math.log10(KMAX + 1), YLO = -10, YHI = Math.log10(peak) + 0.3;
    const X = (k: number) => x0 + (Math.log10(k + 1) / LX) * (x1 - x0);
    const Y = (v: number) => y0 + ((YHI - Math.max(YLO, Math.min(YHI, Math.log10(Math.max(v, 1e-300))))) / (YHI - YLO)) * (y1 - y0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const e of [0, -2, -4, -6, -8, -10]) {
      const y = Y(10 ** e), s = e === 0 ? "1" : `1e${e}`.replace("-", "−");
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText(s, x0 - 6 - ctx.measureText(s).width, y + 4);
    }
    for (const k of [1, 10, 100, 1000, 10000]) {
      const x = X(k), s = k >= 1000 ? `${k / 1000}k` : String(k);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      ctx.fillText(s, x - ctx.measureText(s).width / 2, y1 + 14);
    }
    ctx.fillText("반복 수", x1 - ctx.measureText("반복 수").width, y1 + 26);
    // 허용 오차
    ctx.strokeStyle = C.good; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x0, Y(TOL)); ctx.lineTo(x1, Y(TOL)); ctx.stroke(); ctx.setLineDash([]);
    // 곡선: 화면 x 픽셀마다 한 점 정도만 그린다
    for (const m of METHODS) {
      const h = res[m.key];
      ctx.strokeStyle = col(m.key); ctx.lineWidth = m.key === "cg" ? 2.6 : 2; ctx.lineJoin = "round";
      ctx.beginPath();
      let lastPx = -1;
      for (let k = 0; k < h.length; k++) {
        const x = X(k);
        if (k !== h.length - 1 && x - lastPx < 0.75) continue;
        lastPx = x;
        k ? ctx.lineTo(x, Y(h[k])) : ctx.moveTo(x, Y(h[k]));
      }
      ctx.stroke();
      const kl = h.length - 1;
      ctx.fillStyle = col(m.key); ctx.beginPath(); ctx.arc(X(kl), Y(h[kl]), 4, 0, 7); ctx.fill();
    }
    // 범례
    ctx.font = "600 12px Pretendard, sans-serif";
    let lx = x0;
    for (const m of METHODS) {
      ctx.fillStyle = col(m.key); ctx.fillRect(lx, 11, 12, 3);
      ctx.fillStyle = C.ink; ctx.fillText(m.label, lx + 16, 16);
      lx += 16 + ctx.measureText(m.label).width + 14;
    }
  }

  function update() {
    info = sigmaFor(n, 10 ** lk);
    res = race(n, info.sigma);
    pk.set(lk);
    const it = (h: number[]) => (h[h.length - 1] <= TOL ? String(h.length - 1) : `> ${KMAX}`);
    const c = Math.cos(Math.PI / (n + 1)), rhoJ = (2 * c) / (2 + info.sigma);
    const est = (rho: number) => Math.round(Math.log(TOL) / Math.log(rho));
    ro.set("kappa", `κ = ${kfmt(info.kappa)}${info.sigma === 0 ? " (σ = 0, Poisson)" : ` (σ = ${info.sigma.toPrecision(2)})`}`);
    ro.set("rho", `ρ(Jacobi) = ${rhoJ.toFixed(4)}`);
    ro.set("j", `Jacobi ${it(res.jacobi)}${res.jacobi[res.jacobi.length - 1] > TOL ? ` (≈ ${est(rhoJ)})` : ""}`);
    ro.set("gs", `Gauss–Seidel ${it(res.gs)}${res.gs[res.gs.length - 1] > TOL ? ` (≈ ${est(rhoJ * rhoJ)})` : ""}`);
    ro.set("cg", `CG ${it(res.cg)}`, "hot");
    draw();
  }
  const schedule = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; update(); }); };

  const pn = param({ label: "n", min: 10, max: 200, step: 10, value: n, fmt: (v) => `${v}`, aria: "행렬 크기 n", onInput: (v) => { n = v; schedule(); } });
  const pk = param({ label: "κ", min: 0.3, max: 5, step: 0.05, value: lk, fmt: (v) => kfmt(sigmaFor(n, 10 ** v).kappa), aria: "조건수 κ(로그 눈금, Poisson의 κ가 상한)", onInput: (v) => { lk = v; schedule(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row, pk.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
