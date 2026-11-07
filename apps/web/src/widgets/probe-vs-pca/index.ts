// PCA 시각화 vs 선형 프로브 (data/widgets.toml: probe-vs-pca, 18.8장 "시각화").
// 본문과 같은 감쇠 진동 신호 x(t) = e^{−γt} cos(ωt + φ) + ε를 학습하지 않은 무작위 ReLU 층에 통과시킨 표현에서,
// "빠른 진동인가(ω > 중앙값)"를 판별하는 최소제곱 선형 분류기를 주성분 k개만 남긴 표현으로 학습한다.
// 학습 자료에서 주성분 좌표는 서로 직교하고 평균이 0이므로, 최소제곱 해가 주성분마다 따로 w_i = (P_i·y)/λ_i로 정해진다.
// 본문(N = 2000, 256차원)보다 작게(N = 1500, 128차원) 만들어 브라우저에서 바로 계산한다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

const N = 1500, T = 64, D = 128, NTR = 1100;
export const KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 80, 100, 128];

/** 대칭행렬의 고윳값 분해(순환 Jacobi). A는 n×n 행 우선 배열이고 덮어쓴다. 고윳값 내림차순과 열 고유벡터 V를 돌려준다. */
export function symEig(A: Float64Array, n: number) {
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0, diag = 0;
    for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) { const v = A[p * n + q] ** 2; if (p === q) diag += v; else off += v; }
    if (off < 1e-22 * diag) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const tt = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
      for (let k = 0; k < n; k++) { // 열 p, q 회전
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) { // 행 p, q 회전
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b * n + b] - A[a * n + a]);
  const vals = order.map((i) => A[i * n + i]);
  const vecs = new Float64Array(n * n);
  order.forEach((src, dst) => { for (let k = 0; k < n; k++) vecs[k * n + dst] = V[k * n + src]; });
  return { vals, vecs };
}

/** 자료를 만들고 PCA와 k별 probe를 계산한다. 검산을 위해 원자료도 돌려준다. */
export function compute(seed = 1) {
  const r = rng(seed);
  const ts = Array.from({ length: T }, (_, i) => i / (T - 1));
  const omega = new Float64Array(N), gamma = new Float64Array(N);
  const X = new Float64Array(N * T);
  for (let n = 0; n < N; n++) {
    omega[n] = 2 * Math.PI * (3 + 5 * r.u());          // rad/s, 3–8 Hz
    gamma[n] = 0.5 + 3.5 * r.u();                      // 1/s
    const ph = 2 * Math.PI * r.u();
    for (let i = 0; i < T; i++) X[n * T + i] = Math.exp(-gamma[n] * ts[i]) * Math.cos(omega[n] * ts[i] + ph) + 0.1 * r.normal();
  }
  const W = new Float64Array(T * D);
  for (let i = 0; i < T * D; i++) W[i] = r.normal() / Math.sqrt(T);
  const H = new Float64Array(N * D);                   // 무작위 ReLU 층 (N, D)
  for (let n = 0; n < N; n++) for (let j = 0; j < D; j++) {
    let s = 0;
    for (let i = 0; i < T; i++) s += X[n * T + i] * W[i * D + j];
    H[n * D + j] = s > 0 ? s : 0;
  }
  const sorted = Array.from(omega).sort((a, b) => a - b);
  const med = (sorted[N / 2 - 1] + sorted[N / 2]) / 2;
  const fast = Array.from(omega, (o) => (o > med ? 1 : 0));
  // 학습 자료 평균으로 중심화
  const mean = new Float64Array(D);
  for (let n = 0; n < NTR; n++) for (let j = 0; j < D; j++) mean[j] += H[n * D + j] / NTR;
  const Z = new Float64Array(N * D);
  for (let i = 0; i < N * D; i++) Z[i] = H[i] - mean[i % D];
  const Cv = new Float64Array(D * D);                  // Zᵀ Z (학습 자료)
  for (let n = 0; n < NTR; n++) {
    const o = n * D;
    for (let a = 0; a < D; a++) { const za = Z[o + a]; if (za === 0) continue; for (let b = a; b < D; b++) Cv[a * D + b] += za * Z[o + b]; }
  }
  for (let a = 0; a < D; a++) for (let b = 0; b < a; b++) Cv[a * D + b] = Cv[b * D + a];
  const { vals, vecs } = symEig(Cv, D);
  const total = vals.reduce((s, v) => s + Math.max(v, 0), 0);
  const P = new Float64Array(N * D);                   // 주성분 좌표 (N, D)
  for (let n = 0; n < N; n++) for (let k = 0; k < D; k++) {
    let s = 0;
    for (let j = 0; j < D; j++) s += Z[n * D + j] * vecs[j * D + k];
    P[n * D + k] = s;
  }
  // 최소제곱 probe: 학습 자료에서 주성분 좌표는 직교·평균 0이므로 w_i = (P_i · y) / λ_i, b = mean(y)
  const y = fast.map((v) => 2 * v - 1);
  let b = 0;
  for (let n = 0; n < NTR; n++) b += y[n] / NTR;
  const w = new Float64Array(D);
  for (let k = 0; k < D; k++) { let s = 0; for (let n = 0; n < NTR; n++) s += P[n * D + k] * y[n]; w[k] = s / vals[k]; }
  // k별 test 점수(누적합)와 정확도
  const NTE = N - NTR;
  const score = new Float64Array(NTE).fill(b);
  const acc = new Float64Array(D + 1), evr = new Float64Array(D + 1);
  const scores: Float64Array[] = [new Float64Array(score)];
  for (let k = 1; k <= D; k++) {
    let hit = 0;
    for (let m = 0; m < NTE; m++) { score[m] += w[k - 1] * P[(NTR + m) * D + k - 1]; if ((score[m] > 0 ? 1 : 0) === fast[NTR + m]) hit++; }
    acc[k] = hit / NTE;
    evr[k] = evr[k - 1] + Math.max(vals[k - 1], 0) / total;
    scores.push(new Float64Array(score));
  }
  return { X, W, omega, fast, P, acc, evr, scores, NTE };
}

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 560;
  const f = frame(root, "PCA 시각화 vs 선형 프로브", "manipulate", narrow ? 0.8 : 0.5,
    "시험 표본을 두 주성분 또는 probe 점수로 그린 산점도와, 주성분 수에 따른 probe 정확도·설명 분산 곡선");
  const R = compute();
  let ki = KS.indexOf(2);
  let view: "pca" | "probe" = "pca";
  const k = () => KS[ki];

  // 산점도 축 범위(시험 표본 기준)
  const range = (vals: number[]) => { const s = [...vals].sort((a, b) => a - b); const lo = s[Math.floor(0.01 * s.length)], hi = s[Math.ceil(0.99 * s.length) - 1]; const m = (hi - lo) * 0.08; return [lo - m, hi + m] as const; };
  const te = Array.from({ length: R.NTE }, (_, m) => 1100 + m);
  const pc1 = te.map((n) => R.P[n * D]), pc2 = te.map((n) => R.P[n * D + 1]);
  const r1 = range(pc1), r2 = range(pc2);

  function layout() {
    const W = f.width, H = f.height;
    if (W < 560) {
      const sh = H * 0.56;
      return { sc: { x: 30, y: 22, w: W - 44, h: sh - 34 }, cv: { x: 40, y: sh + 24, w: W - 58, h: H - sh - 52 } };
    }
    const sw = Math.min(W * 0.48, H + 40);
    return { sc: { x: 30, y: 22, w: sw - 40, h: H - 48 }, cv: { x: sw + 34, y: 22, w: W - sw - 52, h: H - 56 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = layout();
    // 산점도
    const g = L.sc;
    const xs = view === "pca" ? pc1 : Array.from(R.scores[k()]);
    const ys = view === "pca" ? pc2 : pc1;
    const rx = view === "pca" ? r1 : range(xs), ry = view === "pca" ? r2 : r1;
    const X = (v: number) => g.x + ((v - rx[0]) / (rx[1] - rx[0])) * g.w;
    const Y = (v: number) => g.y + g.h - ((v - ry[0]) / (ry[1] - ry[0])) * g.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(g.x, g.y, g.w, g.h);
    if (view === "probe" && rx[0] < 0 && rx[1] > 0) { // 판별 경계 점수 = 0
      ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(X(0), g.y); ctx.lineTo(X(0), g.y + g.h); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(g.x, g.y, g.w, g.h); ctx.clip();
    for (let pass = 0; pass < 2; pass++) {
      ctx.fillStyle = pass ? C.edge : C.ink2;
      te.forEach((n, m) => {
        if (R.fast[n] !== pass) return;
        ctx.globalAlpha = 0.7;
        ctx.beginPath(); ctx.arc(X(xs[m]), Y(ys[m]), 2.3, 0, 7); ctx.fill();
      });
    }
    ctx.globalAlpha = 1; ctx.restore();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    const xl = view === "pca" ? "PC1" : `probe 점수 (PC ${k()}개)`;
    ctx.fillText(xl, g.x + g.w - ctx.measureText(xl).width, g.y + g.h + 14);
    ctx.save(); ctx.translate(g.x - 8, g.y + 4); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right";
    ctx.fillText(view === "pca" ? "PC2" : "PC1", 0, 0); ctx.restore();
    // 범례
    ctx.font = "11px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(g.x + 8, g.y - 10, 4, 0, 7); ctx.fill();
    ctx.fillStyle = C.ink; ctx.fillText("빠른 진동", g.x + 16, g.y - 6);
    ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(g.x + 86, g.y - 10, 4, 0, 7); ctx.fill();
    ctx.fillStyle = C.ink; ctx.fillText("느린 진동", g.x + 94, g.y - 6);

    // 곡선: k(로그 축)에 따른 정확도·설명 분산
    const c = L.cv;
    const KX = (kk: number) => c.x + (Math.log(kk) / Math.log(D)) * c.w;
    const VY = (v: number) => c.y + c.h - ((v - 0.3) / 0.7) * c.h;
    ctx.strokeStyle = C.faint; ctx.strokeRect(c.x, c.y, c.w, c.h);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const kk of [1, 2, 4, 8, 16, 32, 64, 128]) { const t = String(kk); ctx.fillText(t, KX(kk) - ctx.measureText(t).width / 2, c.y + c.h + 14); }
    for (const v of [0.5, 1.0]) ctx.fillText(v.toFixed(1), c.x - 26, VY(v) + 4);
    const kt = "주성분 수 k (로그 축)";
    ctx.fillText(kt, c.x + c.w - ctx.measureText(kt).width, c.y - 6);
    ctx.strokeStyle = C.grid2; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(c.x, VY(0.5)); ctx.lineTo(c.x + c.w, VY(0.5)); ctx.stroke(); ctx.setLineDash([]);
    const curve = (arr: Float64Array, color: string, lw: number) => {
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
      for (let kk = 1; kk <= D; kk++) { const px = KX(kk), py = VY(Math.max(0.3, arr[kk])); if (kk === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
      ctx.stroke(); ctx.lineWidth = 1;
    };
    curve(R.evr, C.ink2, 2);
    curve(R.acc, C.edge, 2.5);
    ctx.strokeStyle = C.mid; ctx.beginPath(); ctx.moveTo(KX(k()), c.y); ctx.lineTo(KX(k()), c.y + c.h); ctx.stroke();
    ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(KX(k()), VY(Math.max(0.3, R.evr[k()])), 4.5, 0, 7); ctx.fill();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(KX(k()), VY(Math.max(0.3, R.acc[k()])), 5.5, 0, 7); ctx.fill();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.textAlign = "right";
    ctx.fillStyle = C.edge; ctx.fillText("probe 정확도", c.x + c.w - 6, c.y + c.h - 24);
    ctx.fillStyle = C.ink2; ctx.fillText("설명 분산", c.x + c.w - 6, c.y + c.h - 8);
    ctx.textAlign = "left";
  }

  function update() {
    ro.set("k", `PC ${k()}개 · 설명 분산 ${R.evr[k()].toFixed(3)}`);
    ro.set("acc", `빠른 진동 판별 정확도 ${R.acc[k()].toFixed(3)}`, R.acc[k()] < 0.6 ? "hot" : "ok");
    ro.set("ref", `PC 2개 ${R.acc[2].toFixed(3)} · 전체 ${D}개 ${R.acc[D].toFixed(3)}`);
    draw();
  }

  const pv = presets<"pca" | "probe">([{ key: "pca", label: "PC1·PC2 그림" }, { key: "probe", label: "probe 점수로 보기" }], (v) => { view = v; update(); }, view);
  pv.row.classList.add("segmented");
  const pk = param({ label: "k", min: 0, max: KS.length - 1, step: 1, value: ki, fmt: (i) => String(KS[i]), aria: "probe에 남기는 주성분 수", onInput: (i) => { ki = i; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pk.row);
  const ro = readouts();
  f.panel.append(pv.row, grid, ro.row);
  f.onResize(draw);
  update();
}
