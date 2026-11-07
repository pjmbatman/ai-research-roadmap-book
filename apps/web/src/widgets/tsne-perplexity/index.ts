// t-SNE perplexity (data/widgets.toml: tsne-perplexity).
// 16.5장의 세 군집을 작게 만든다(군집마다 80점, 5차원, 이 위젯의 고정 시드):
//   A ~ N(0, 0.2²I), B ~ N((6, 0, 0, 0, 0), 2²I) — A보다 10배 퍼짐, C ~ N((40, 0, 0, 0, 0), 0.2²I) — 멀리 떨어짐
// 미리 계산하지 않고 브라우저에서 정확한(exact) t-SNE를 돌린다(scikit-learn 기본값과 같은 설정):
//   p_{j|i}: 점마다 이분 탐색으로 엔트로피 = ln(perplexity), P = (P + Pᵀ)/Σ, 2차원은 Student t(자유도 1)
//   ∂KL/∂y_i = 4 Σ_j (p_ij − q_ij)(y_i − y_j)/(1 + ‖y_i − y_j‖²)
//   초기값 PCA(첫 성분 표준편차 10⁻⁴), 학습률 max(N/12/4, 50), 처음 250번은 P×12·momentum 0.5, 이후 0.8, gain 갱신, 모두 1000번
// 읽음값의 퍼짐 = 군집 중심까지 평균 거리, 거리 = 군집 중심 사이 거리(본문 셀과 같은 정의).
import { frame, param, readouts, el, C, rng, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

export const NC = 80, N = 3 * NC, DIM = 5, ITER = 1000, EXAG_IT = 250;
export const DATA = (() => {
  const r = rng(1605);
  const pts: number[][] = [];
  const blob = (s: number, c0: number) => { for (let i = 0; i < NC; i++) pts.push(Array.from({ length: DIM }, (_, d) => s * r.normal() + (d === 0 ? c0 : 0))); };
  blob(0.2, 0); blob(2.0, 6); blob(0.2, 40);
  return pts;
})();
const D2 = (() => { const d = new Float64Array(N * N); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < DIM; k++) s += (DATA[i][k] - DATA[j][k]) ** 2; d[i * N + j] = s; } return d; })();

/** 군집 퍼짐 비(B/A)와 중심 거리 비(AC/AB). Z는 N×dim 행 우선 배열 */
export function ratios(Z: ArrayLike<number>, dim: number) {
  const cen = [0, 1, 2].map((g) => Array.from({ length: dim }, (_, d) => { let s = 0; for (let i = g * NC; i < (g + 1) * NC; i++) s += Z[i * dim + d]; return s / NC; }));
  const spread = (g: number) => { let s = 0; for (let i = g * NC; i < (g + 1) * NC; i++) { let q = 0; for (let d = 0; d < dim; d++) q += (Z[i * dim + d] - cen[g][d]) ** 2; s += Math.sqrt(q); } return s / NC; };
  const gap = (a: number, b: number) => Math.sqrt(cen[a].reduce((s, v, d) => s + (v - cen[b][d]) ** 2, 0));
  return { spread: spread(1) / spread(0), gap: gap(0, 2) / gap(0, 1) };
}
export const ORIG = ratios(DATA.flat(), DIM);

/** perplexity로 대칭 결합확률 P를 만든다 */
export function affinities(perp: number) {
  const P = new Float64Array(N * N), target = Math.log(perp);
  for (let i = 0; i < N; i++) {
    let beta = 1, lo = -Infinity, hi = Infinity;
    for (let it = 0; it < 100; it++) {
      let sum = 0, sdp = 0;
      for (let j = 0; j < N; j++) { if (j === i) continue; const p = Math.exp(-D2[i * N + j] * beta); P[i * N + j] = p; sum += p; sdp += D2[i * N + j] * p; }
      sum = Math.max(sum, 1e-300);
      const H = Math.log(sum) + (beta * sdp) / sum;
      for (let j = 0; j < N; j++) if (j !== i) P[i * N + j] /= sum;
      const diff = H - target;
      if (Math.abs(diff) < 1e-5) break;
      if (diff > 0) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
      else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
    }
  }
  const S = new Float64Array(N * N);
  let tot = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { S[i * N + j] = P[i * N + j] + P[j * N + i]; tot += S[i * N + j]; }
  for (let k = 0; k < N * N; k++) S[k] = Math.max(S[k] / tot, 2.220446049250313e-16);
  return S;
}

/** PCA 초기값: 공분산의 첫 두 고유벡터(거듭제곱법), 첫 성분 표준편차를 1e-4로 */
function pcaInit() {
  const mu = Array.from({ length: DIM }, (_, d) => DATA.reduce((s, x) => s + x[d], 0) / N);
  const Cv = Array.from({ length: DIM }, (_, a) => Array.from({ length: DIM }, (_, b) => DATA.reduce((s, x) => s + (x[a] - mu[a]) * (x[b] - mu[b]), 0) / N));
  const vecs: number[][] = [];
  for (let k = 0; k < 2; k++) {
    let v: number[] = Array.from({ length: DIM }, (_, d) => (d === k ? 1 : 0.1));
    for (let it = 0; it < 500; it++) {
      let w = Cv.map((row) => row.reduce((s, c, d) => s + c * v[d], 0));
      for (const u of vecs) { const dot = w.reduce((s, x, d) => s + x * u[d], 0); w = w.map((x, d) => x - dot * u[d]); }
      const nrm = Math.hypot(...w); v = w.map((x) => x / nrm);
    }
    vecs.push(v);
  }
  const Y = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) for (let k = 0; k < 2; k++) Y[i * 2 + k] = DATA[i].reduce((s, x, d) => s + (x - mu[d]) * vecs[k][d], 0);
  const m0 = Y.filter((_, i) => i % 2 === 0).reduce((s, v) => s + v, 0) / N;
  const sd = Math.sqrt(Y.filter((_, i) => i % 2 === 0).reduce((s, v) => s + (v - m0) ** 2, 0) / N);
  return Y.map((v) => (v / sd) * 1e-4);
}
const Y0 = pcaInit();

/** 한 perplexity의 최적화 상태. step(k)로 k번 반복한다 */
export function tsne(perp: number) {
  const P = affinities(perp);
  const Y = Float64Array.from(Y0), upd = new Float64Array(N * 2), gains = new Float64Array(N * 2).fill(1), G = new Float64Array(N * 2);
  const num = new Float64Array(N * N);
  const lr = Math.max(N / 12 / 4, 50);
  let it = 0, kl = NaN;
  const step = (k: number) => {
    for (let s = 0; s < k && it < ITER; s++, it++) {
      const ex = it < EXAG_IT ? 12 : 1, mom = it < EXAG_IT ? 0.5 : 0.8;
      let Z = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1]; const q = 1 / (1 + dx * dx + dy * dy); num[i * N + j] = q; num[j * N + i] = q; Z += 2 * q; }
      G.fill(0);
      let klv = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const q = num[i * N + j], qq = Math.max(q / Z, 1e-12), p = ex * P[i * N + j];
        const c = 4 * (p - qq) * q;
        G[2 * i] += c * (Y[2 * i] - Y[2 * j]); G[2 * i + 1] += c * (Y[2 * i + 1] - Y[2 * j + 1]);
        if (ex === 1) klv += P[i * N + j] * Math.log(P[i * N + j] / qq);
      }
      if (ex === 1) kl = klv;
      for (let k2 = 0; k2 < 2 * N; k2++) {
        gains[k2] = upd[k2] * G[k2] < 0 ? gains[k2] + 0.2 : gains[k2] * 0.8;
        if (gains[k2] < 0.01) gains[k2] = 0.01;
        upd[k2] = mom * upd[k2] - lr * gains[k2] * G[k2];
        Y[k2] += upd[k2];
      }
    }
    return it;
  };
  return { Y, step, get it() { return it; }, get kl() { return kl; } };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "t-SNE perplexity", "manipulate", 0.6, "세 군집 A(촘촘), B(10배 퍼짐), C(멀리)의 t-SNE 2차원 좌표. perplexity를 바꾸면 처음부터 다시 최적화한다");
  let perp = 30;
  let run = tsne(perp);
  const cols = () => [C.edge, C.ink2, C.good];
  const names = ["A 촘촘", "B 10배 퍼짐", "C 멀리"];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const Y = run.Y;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < N; i++) { x0 = Math.min(x0, Y[2 * i]); x1 = Math.max(x1, Y[2 * i]); y0 = Math.min(y0, Y[2 * i + 1]); y1 = Math.max(y1, Y[2 * i + 1]); }
    const pad = 18, top = 28;
    const s = Math.min((W - 2 * pad) / (x1 - x0 || 1), (H - top - pad) / (y1 - y0 || 1));
    const ox = (W - s * (x1 - x0)) / 2, oy = top + (H - top - pad - s * (y1 - y0)) / 2;
    const cs = cols();
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = cs[Math.floor(i / NC)];
      ctx.beginPath(); ctx.arc(ox + (Y[2 * i] - x0) * s, oy + (Y[2 * i + 1] - y0) * s, 3, 0, 7); ctx.fill();
    }
    ctx.font = "11px JetBrains Mono, monospace";
    let lx = pad;
    names.forEach((nm, g) => {
      ctx.fillStyle = cs[g]; ctx.beginPath(); ctx.arc(lx + 4, 14, 4, 0, 7); ctx.fill();
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 12, 18);
      lx += 24 + ctx.measureText(nm).width;
    });
  }

  let raf = 0;
  function readout() {
    const r = ratios(run.Y, 2);
    ro.set("orig", `원공간: 퍼짐 B/A ${ORIG.spread.toFixed(1)}배 · 거리 AC/AB ${ORIG.gap.toFixed(1)}배`);
    ro.set("emb", `t-SNE: 퍼짐 B/A ${r.spread.toFixed(1)}배 · 거리 AC/AB ${r.gap.toFixed(1)}배`, run.it >= ITER ? "hot" : "");
    ro.set("kl", run.it > EXAG_IT ? `반복 ${run.it}/${ITER} · KL ${run.kl.toFixed(3)}` : `반복 ${run.it}/${ITER} · 초기 과장 단계`);
    pageState.set("tsne-perplexity", { perplexity: perp, spread: r.spread, gap: r.gap, done: run.it >= ITER }, root);
  }
  function loop() {
    raf = 0;
    run.step(reducedMotion() ? ITER : 25);
    draw();
    readout();
    if (run.it < ITER) raf = requestAnimationFrame(loop);
  }
  function restart() {
    cancelAnimationFrame(raf);
    run = tsne(perp);
    raf = requestAnimationFrame(loop);
  }

  const pp = param({ label: "perp", min: 2, max: 80, step: 1, value: perp, fmt: (v) => String(v), aria: "perplexity (점마다 고려하는 유효 이웃 수)", onInput: (v) => { perp = v; restart(); } });
  const grid = el("div", "param-grid");
  grid.append(pp.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  readout();
  restart();
}
