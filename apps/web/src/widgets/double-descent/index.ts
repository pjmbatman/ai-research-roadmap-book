// double descent (data/widgets.toml: double-descent). 14.5장의 최소 노름 회귀 실험을 그대로 쓴다.
// 학습 표본 N = 40, 후보 특징 D = 200, 잡음 표준편차 0.1, 참 계수 β_j ∝ j^(−1/2)(‖β‖ = 1). 모델은 앞의 p개 특징만 쓴다.
// λ = 0이면 p ≤ N에서 최소제곱 해, p > N에서 최소 노름 보간 해(X⁺y)이고, λ > 0이면 ridge 해다.
// 입력이 등방 가우시안이라 시험 위험(MSE)은 ‖w − β_p‖² + ‖β_나머지‖² + σ²로 정확히 계산된다. 20번 반복의 중앙값을 그린다.
import { frame, param, readouts, el, C, rng } from "../frame";

export const N = 40, D = 200, NOISE = 0.1, REPS = 20;
export const LAMS = [0, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, 10];
export const BETA = (() => { const b = Array.from({ length: D }, (_, j) => (j + 1) ** -0.5); const s = Math.hypot(...b); return b.map((v) => v / s); })();
/** 반복마다 X (N, D) 행 우선과 y (N,) */
export const DATA = Array.from({ length: REPS }, (_, k) => {
  const r = rng(4242 + k);
  const X = new Float64Array(N * D);
  for (let i = 0; i < N * D; i++) X[i] = r.normal();
  const y = Float64Array.from({ length: N }, (_, i) => { let s = 0; for (let j = 0; j < D; j++) s += X[i * D + j] * BETA[j]; return s + NOISE * r.normal(); });
  return { X, y };
});
const TAIL = (() => { const t = new Float64Array(D + 1); for (let p = D - 1; p >= 0; p--) t[p] = t[p + 1] + BETA[p] ** 2; return t; })();

/** 대칭 양의 정부호 m×m 행렬 A(행 우선, 덮어씀)로 A z = b를 Cholesky로 푼다. */
function cholSolve(A: Float64Array, m: number, b: Float64Array): Float64Array {
  for (let j = 0; j < m; j++) {
    let s = A[j * m + j];
    for (let k = 0; k < j; k++) s -= A[j * m + k] ** 2;
    const d = Math.sqrt(Math.max(s, 1e-300));
    A[j * m + j] = d;
    for (let i = j + 1; i < m; i++) {
      let t = A[i * m + j];
      for (let k = 0; k < j; k++) t -= A[i * m + k] * A[j * m + k];
      A[i * m + j] = t / d;
    }
  }
  const z = Float64Array.from(b);
  for (let i = 0; i < m; i++) { let t = z[i]; for (let k = 0; k < i; k++) t -= A[i * m + k] * z[k]; z[i] = t / A[i * m + i]; }
  for (let i = m - 1; i >= 0; i--) { let t = z[i]; for (let k = i + 1; k < m; k++) t -= A[k * m + i] * z[k]; z[i] = t / A[i * m + i]; }
  return z;
}

/** 반복 rep에서 p = 1..D마다 앞의 p개 특징, ridge λ의 해를 구해 시험 위험·‖w‖·학습 MSE를 낸다. */
export function repCurve(rep: number, lam: number) {
  const { X, y } = DATA[rep];
  const col = (j: number, i: number) => X[i * D + j];
  // p ≤ N: (X_pᵀX_p + λI) w = X_pᵀ y. 앞 N개 특징의 그람 행렬을 한 번 만들어 잘라 쓴다.
  const A = new Float64Array(N * N), Xty = new Float64Array(N);
  for (let a = 0; a < N; a++) {
    for (let c = a; c < N; c++) { let s = 0; for (let i = 0; i < N; i++) s += col(a, i) * col(c, i); A[a * N + c] = A[c * N + a] = s; }
    let s = 0; for (let i = 0; i < N; i++) s += col(a, i) * y[i]; Xty[a] = s;
  }
  // p > N: w = X_pᵀ (X_p X_pᵀ + λI)⁻¹ y (λ = 0이면 최소 노름 보간 해 X_p⁺ y). X_p X_pᵀ는 특징을 하나씩 더하며 갱신한다.
  const G = new Float64Array(N * N);
  const out = { risk: new Float64Array(D), norm: new Float64Array(D), train: new Float64Array(D) };
  for (let p = 1; p <= D; p++) {
    const j = p - 1;
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) G[i * N + k] += col(j, i) * col(j, k);
    let w: Float64Array;
    if (p <= N) {
      const M = new Float64Array(p * p);
      for (let a = 0; a < p; a++) for (let c = 0; c < p; c++) M[a * p + c] = A[a * N + c] + (a === c ? lam : 0);
      w = cholSolve(M, p, Xty.subarray(0, p));
    } else {
      const M = Float64Array.from(G);
      for (let i = 0; i < N; i++) M[i * N + i] += lam;
      const alpha = cholSolve(M, N, y);
      w = new Float64Array(p);
      for (let jj = 0; jj < p; jj++) { let s = 0; for (let i = 0; i < N; i++) s += col(jj, i) * alpha[i]; w[jj] = s; }
    }
    let err = 0, nrm = 0, tr = 0;
    for (let jj = 0; jj < p; jj++) { err += (w[jj] - BETA[jj]) ** 2; nrm += w[jj] ** 2; }
    for (let i = 0; i < N; i++) { let s = 0; for (let jj = 0; jj < p; jj++) s += X[i * D + jj] * w[jj]; tr += (s - y[i]) ** 2; }
    out.risk[j] = err + TAIL[p] + NOISE ** 2; out.norm[j] = Math.sqrt(nrm); out.train[j] = tr / N;
  }
  return out;
}
const median = (a: number[]) => { const s = [...a].sort((u, v) => u - v), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
/** p = 1..D에서 20번 반복의 중앙값 */
export function curves(lam: number) {
  const reps = Array.from({ length: REPS }, (_, k) => repCurve(k, lam));
  const med = (key: "risk" | "norm" | "train") => Array.from({ length: D }, (_, j) => median(reps.map((r) => r[key][j])));
  return { risk: med("risk"), norm: med("norm"), train: med("train") };
}
// 곡선은 결정적이므로 λ마다 한 번만 계산한다(초기화로 다시 그려도 재사용).
const cache = new Map<number, ReturnType<typeof curves>>();
const get = (i: number) => { let c = cache.get(i); if (!c) { c = curves(LAMS[i]); cache.set(i, c); } return c; };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "double descent", "manipulate", narrow ? 0.85 : 0.56, "특징 수 p에 따른 시험 위험 곡선(로그 축)과 보간 임계점 p = N");
  let p = 40, li = 0;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const cur = get(li), base = get(0);
    const x0 = 44, x1 = W - 14, y0 = 16, y1 = H - 28;
    const LO = Math.log10(0.3), HI = Math.log10(100);
    const X = (q: number) => x0 + ((q - 1) / (D - 1)) * (x1 - x0);
    const Y = (v: number) => y1 - ((Math.min(Math.max(Math.log10(v), LO), HI) - LO) / (HI - LO)) * (y1 - y0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const v of [0.3, 1, 3, 10, 30, 100]) {
      const s = String(v);
      ctx.fillText(s, x0 - 6 - ctx.measureText(s).width, Y(v) + 4);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
    }
    for (const q of [1, 40, 100, 150, 200]) { const s = String(q); ctx.fillText(s, X(q) - ctx.measureText(s).width / 2, y1 + 14); }
    ctx.fillText("p", x1 - 8, y1 - 6);
    // 보간 임계점 p = N
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(N), y0); ctx.lineTo(X(N), y1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("p = N", X(N) + 5, y1 - 8);
    // 잡음 바닥 σ²는 0.01이라 축 밖이다. λ > 0이면 최소 노름 곡선을 흐리게 함께 그린다.
    const line = (a: number[], col: string, w: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); a.forEach((v, i) => (i ? ctx.lineTo(X(i + 1), Y(v)) : ctx.moveTo(X(i + 1), Y(v)))); ctx.stroke(); ctx.setLineDash([]);
    };
    if (li > 0) line(base.risk, C.mid, 1.2, [5, 4]);
    line(cur.risk, C.edge, 2.2);
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(X(p), Y(cur.risk[p - 1]), 5, 0, 7); ctx.fill();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const lab = li === 0 ? "시험 위험 (최소 노름)" : `시험 위험 (ridge λ = ${LAMS[li]})`;
    ctx.fillStyle = C.edge; ctx.fillText(lab, x1 - ctx.measureText(lab).width, y0 + 10);
  }

  function update() {
    const c = get(li);
    const r = c.risk[p - 1];
    let bi = 0; c.risk.forEach((v, i) => { if (v < c.risk[bi]) bi = i; });
    ro.set("reg", p < N ? "p < N · 고전 영역" : p === N ? "p = N · 보간 임계점" : "p > N · 과매개변수화", p === N ? "hot" : "");
    ro.set("risk", `시험 위험 ${r.toFixed(2)}`, "hot");
    ro.set("norm", `‖w‖ ${c.norm[p - 1].toFixed(2)}`);
    ro.set("train", `학습 MSE ${c.train[p - 1].toFixed(3)}`);
    ro.set("best", `최소 ${c.risk[bi].toFixed(2)} @ p = ${bi + 1}`);
    draw();
  }

  let raf = 0;
  const later = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(update); };
  const pp = param({ label: "p", min: 1, max: D, step: 1, value: p, fmt: (v) => `${v}`, aria: "모델이 쓰는 특징 수 p", onInput: (v) => { p = v; later(); } });
  const pl = param({ label: "λ", min: 0, max: LAMS.length - 1, step: 1, value: li, fmt: (v) => (LAMS[v] === 0 ? "0" : String(LAMS[v])), aria: "ridge 정규화 세기 λ", onInput: (v) => { li = v; later(); } });
  pl.input.setAttribute("aria-valuetext", "0");
  pl.input.addEventListener("input", () => pl.input.setAttribute("aria-valuetext", String(LAMS[li])));
  const grid = el("div", "param-grid");
  grid.append(pp.row, pl.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
  // 나머지 λ의 곡선은 한가할 때 미리 계산해 둔다(한 λ에 0.1~0.3초).
  let k = 1;
  const warm = () => { if (k >= LAMS.length || !root.isConnected) return; get(k++); setTimeout(warm, 30); };
  setTimeout(warm, 200);
}
