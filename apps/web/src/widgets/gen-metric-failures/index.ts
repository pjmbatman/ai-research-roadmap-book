// 생성 지표의 실패 (data/widgets.toml: gen-metric-failures, 19.7장 "평가 축"·"지표").
// 본문 셀과 같은 2차원 혼합 가우시안(중심 (±2, ±2), 표준편차 0.4)과 다섯 "생성모델":
//   좋은 모델, 모드 하나 누락, 흐릿함(std 2배), 학습 자료 복사(+0.02 잡음), 평균·공분산만 맞춘 가우시안.
// 지표는 모두 held-out 실제 자료(test) 기준이다.
//   precision·recall  k = 3 최근접 이웃 공의 합집합(Kynkäänniemi 외, 2019)
//   학습 거리 비       median d(생성, train) / median d(test, train)
//   FID               두 특징 집합에 가우시안을 맞춘 Fréchet 거리. 원 좌표(2차원)와 무작위 ReLU 특징 32개
//   MMD²              RBF kernel의 불편 추정량. 대역폭 = 거리의 중앙값, 또는 0.5
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";

type Pt = [number, number];
const MODES: Pt[] = [[2, 2], [-2, 2], [-2, -2], [2, -2]];
export type Model = "good" | "drop" | "blur" | "copy" | "gauss";
export const MODELS: { key: Model; label: string; short: string }[] = [
  { key: "good", label: "좋은 모델", short: "좋음" },
  { key: "drop", label: "모드 누락", short: "모드 누락" },
  { key: "blur", label: "흐릿함", short: "흐릿함" },
  { key: "copy", label: "학습 자료 복사", short: "복사" },
  { key: "gauss", label: "가우시안 하나", short: "가우시안" },
];

const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
function quickselect(arr: Float64Array, k: number) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const pivot = arr[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) { while (arr[i] < pivot) i++; while (arr[j] > pivot) j--; if (i <= j) { const t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; } }
    if (k <= j) hi = j; else if (k >= i) lo = i; else return arr[k];
  }
  return arr[k];
}
const median = (a: Float64Array) => { const c = Float64Array.from(a); const n = c.length; return n % 2 ? quickselect(c, (n - 1) / 2) : 0.5 * (quickselect(Float64Array.from(c), n / 2 - 1) + quickselect(c, n / 2)); };

/** 각 점의 k번째 이웃 거리(자기 자신 제외). 가장 가까운 k개만 삽입 정렬로 유지한다. */
function kthRadius(X: Pt[], k: number) {
  const best = new Float64Array(k);
  return X.map((p, i) => {
    best.fill(Infinity);
    for (let j = 0; j < X.length; j++) {
      if (j === i) continue;
      const v = d2(p, X[j]);
      if (v >= best[k - 1]) continue;
      let m = k - 1;
      while (m > 0 && best[m - 1] > v) { best[m] = best[m - 1]; m--; }
      best[m] = v;
    }
    return Math.sqrt(best[k - 1]);
  });
}
export function precisionRecall(real: Pt[], fake: Pt[], k = 3) {
  const rr = kthRadius(real, k), rf = kthRadius(fake, k);
  const prec = fake.filter((p) => real.some((q, j) => d2(p, q) <= rr[j] ** 2)).length / fake.length;
  const rec = real.filter((p) => fake.some((q, j) => d2(p, q) <= rf[j] ** 2)).length / real.length;
  return { prec, rec };
}
export const nnMedian = (A: Pt[], train: Pt[]) => median(Float64Array.from(A, (p) => { let m = Infinity; for (const q of train) m = Math.min(m, d2(p, q)); return Math.sqrt(m); }));

// 대칭행렬 고윳값(순환 Jacobi)
function eigSym(A0: number[][]) {
  const n = A0.length, A = A0.map((r) => r.slice()), V: number[][] = A0.map((_, i) => A0.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0, tot = 0;
    for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) { tot += A[p][q] ** 2; if (p !== q) off += A[p][q] ** 2; }
    if (off <= 1e-24 * tot) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
    }
  }
  return { vals: A.map((r, i) => r[i]), V };
}
function meanCov(F: number[][]) {
  const n = F.length, d = F[0].length, mu = new Array(d).fill(0);
  F.forEach((r) => r.forEach((v, j) => (mu[j] += v / n)));
  const S = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of F) for (let a = 0; a < d; a++) { const da = r[a] - mu[a]; for (let b = a; b < d; b++) S[a][b] += (da * (r[b] - mu[b])) / (n - 1); }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) S[a][b] = S[b][a];
  return { mu, S };
}
/** Fréchet 거리 ‖μa − μb‖² + tr(Σa + Σb − 2(Σa Σb)^{1/2}). tr((ΣaΣb)^{1/2}) = Σ √eig(Σa^{1/2} Σb Σa^{1/2}) */
export function frechet(A: number[][], Bm: number[][]) {
  const a = meanCov(A), b = meanCov(Bm), d = a.mu.length;
  const { vals, V } = eigSym(a.S);
  const half = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => { let s = 0; for (let k = 0; k < d; k++) s += V[i][k] * Math.sqrt(Math.max(vals[k], 0)) * V[j][k]; return s; }));
  const mul = (X: number[][], Y: number[][]) => X.map((r) => Y[0].map((_, j) => r.reduce((s, v, k) => s + v * Y[k][j], 0)));
  const Mx = mul(mul(half, b.S), half);
  const tr = eigSym(Mx).vals.reduce((s, v) => s + Math.sqrt(Math.max(v, 0)), 0);
  let m2 = 0, trs = 0;
  for (let i = 0; i < d; i++) { m2 += (a.mu[i] - b.mu[i]) ** 2; trs += a.S[i][i] + b.S[i][i]; }
  return m2 + trs - 2 * tr;
}
/** 불편 MMD²(RBF). 대역폭 목록에 undefined가 있으면 합친 표본의 쌍 거리 중앙값을 쓴다. */
export function mmd2(X: Pt[], Y: Pt[], bws: (number | undefined)[] = [undefined]) {
  const Z = X.concat(Y), n = X.length, m = Y.length, N = Z.length;
  const D = new Float64Array((N * (N - 1)) / 2);
  let c = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) D[c++] = d2(Z[i], Z[j]);
  let med = NaN;
  return bws.map((bw0) => {
    if (bw0 === undefined && Number.isNaN(med)) med = Math.sqrt(median(D));
    const bw = bw0 ?? med, g = 1 / (2 * bw * bw);
    let kxx = 0, kyy = 0, kxy = 0, q = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const k = Math.exp(-D[q++] * g);
      if (j < n) kxx += k; else if (i >= n) kyy += k; else kxy += k;
    }
    return { mmd: (2 * kxx) / (n * (n - 1)) + (2 * kyy) / (m * (m - 1)) - (2 * kxy) / (n * m), bw };
  });
}

export function makeWorld(seed: number) {
  const r = rng(seed * 104729 + 7);
  const real = (n: number, std = 0.4, use = 4): Pt[] => Array.from({ length: n }, () => { const k = Math.min(use - 1, Math.floor(r.u() * use)); return [MODES[k][0] + std * r.normal(), MODES[k][1] + std * r.normal()]; });
  const train = real(500), test = real(500);
  const Wf = [Array.from({ length: 32 }, () => r.normal()), Array.from({ length: 32 }, () => r.normal())];
  const feat = (X: Pt[]) => X.map((p) => Wf[0].map((w, j) => Math.max(0, p[0] * w + p[1] * Wf[1][j] + 0.5)));
  // 학습 자료에 평균·공분산을 맞춘 가우시안
  const tr = meanCov(train);
  const L00 = Math.sqrt(tr.S[0][0]), L10 = tr.S[1][0] / L00, L11 = Math.sqrt(tr.S[1][1] - L10 * L10);
  const gen: Record<Model, (n: number) => Pt[]> = {
    good: (n) => real(n), drop: (n) => real(n, 0.4, 3), blur: (n) => real(n, 0.8),
    copy: (n) => train.slice(0, n).map((p) => [p[0] + 0.02 * r.normal(), p[1] + 0.02 * r.normal()] as Pt),
    gauss: (n) => Array.from({ length: n }, () => { const a = r.normal(), b = r.normal(); return [tr.mu[0] + L00 * a, tr.mu[1] + L10 * a + L11 * b] as Pt; }),
  };
  return { train, test, feat, gen };
}

export type Row = { prec: number; rec: number; ratio: number; fidRaw: number; fidRelu: number; mmdMed: number; mmd05: number };
export function evaluate(w: ReturnType<typeof makeWorld>, n: number) {
  const test = w.test.slice(0, n);
  const base = nnMedian(w.test, w.train);
  const fakes = {} as Record<Model, Pt[]>, rows = {} as Record<Model, Row>;
  const ftest = w.feat(test);
  for (const { key } of MODELS) {
    const fk = w.gen[key](n);
    fakes[key] = fk;
    const { prec, rec } = precisionRecall(test, fk);
    rows[key] = {
      prec, rec, ratio: nnMedian(fk, w.train) / base,
      fidRaw: frechet(fk, test), fidRelu: frechet(w.feat(fk), ftest),
      ...(([a, b]) => ({ mmdMed: a.mmd, mmd05: b.mmd }))(mmd2(fk, test, [undefined, 0.5])),
    };
  }
  return { fakes, rows, test };
}

const ROWS: { key: keyof Row; label: string; fmt: (v: number) => string }[] = [
  { key: "prec", label: "precision", fmt: (v) => v.toFixed(2) },
  { key: "rec", label: "recall", fmt: (v) => v.toFixed(2) },
  { key: "ratio", label: "학습 거리 비", fmt: (v) => v.toFixed(2) },
  { key: "fidRaw", label: "FID 원 좌표", fmt: (v) => v.toFixed(2) },
  { key: "fidRelu", label: "FID ReLU 32", fmt: (v) => v.toFixed(1) },
  { key: "mmdMed", label: "MMD² h=중앙값", fmt: (v) => v.toFixed(4) },
  { key: "mmd05", label: "MMD² h=0.5", fmt: (v) => v.toFixed(4) },
];

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 560;
  const f = frame(root, "생성 지표의 실패", "manipulate", narrow ? 0.9 : 0.5,
    "실제 자료와 생성 표본의 산점도, 그리고 다섯 모델의 지표를 비교한 표");
  let model: Model = "drop", n = 500, seed = 1;
  let world = makeWorld(seed);
  let res = evaluate(world, n);

  function layout() {
    const W = f.width, H = f.height;
    if (W < 560) { const s = Math.min(W - 40, H * 0.44); return { sc: { x: (W - s) / 2, y: 22, s }, tb: { x: 2, y: s + 28, w: W - 4, h: H - s - 30 } }; }
    const s = Math.min(H - 36, W * 0.42);
    return { sc: { x: 12, y: 22, s }, tb: { x: s + 30, y: 14, w: W - s - 36, h: H - 20 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { sc, tb } = layout();
    const B = 4.2, P = (p: Pt): Pt => [sc.x + ((p[0] + B) / (2 * B)) * sc.s, sc.y + ((B - p[1]) / (2 * B)) * sc.s];
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(sc.x, sc.y, sc.s, sc.s);
    ctx.save(); ctx.beginPath(); ctx.rect(sc.x, sc.y, sc.s, sc.s); ctx.clip();
    ctx.fillStyle = alpha(C.mid, 0.55);
    for (const p of res.test) { const q = P(p); ctx.beginPath(); ctx.arc(q[0], q[1], 2, 0, 7); ctx.fill(); }
    if (model === "copy") { // 복사 모델: 학습 자료(테두리)와 거의 겹친다
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      for (const p of world.train.slice(0, n)) { const q = P(p); ctx.beginPath(); ctx.arc(q[0], q[1], 3.6, 0, 7); ctx.stroke(); }
    }
    ctx.fillStyle = alpha(C.edge, 0.85);
    for (const p of res.fakes[model]) { const q = P(p); ctx.beginPath(); ctx.arc(q[0], q[1], 2.2, 0, 7); ctx.fill(); }
    ctx.restore();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid; ctx.fillText("실제(test)", sc.x, sc.y - 6);
    ctx.fillStyle = C.edge; ctx.fillText(MODELS.find((m) => m.key === model)!.label, sc.x + 66, sc.y - 6);
    if (model === "copy") { ctx.fillStyle = C.ink2; ctx.fillText("○ 학습 자료", sc.x + sc.s - 70, sc.y - 6); }

    // 지표 표: 행 = 지표, 열 = 모델. 막대는 행마다 최댓값으로 정규화
    const lw = Math.min(104, tb.w * 0.27), cw = (tb.w - lw) / MODELS.length;
    const rh = Math.min(30, (tb.h - 22) / ROWS.length);
    ctx.font = `${cw < 64 ? 10 : 11}px Pretendard Variable, sans-serif`;
    MODELS.forEach((m, j) => {
      const x = tb.x + lw + j * cw;
      if (m.key === model) { ctx.fillStyle = alpha(C.edge, 0.1); ctx.fillRect(x + 1, tb.y, cw - 2, 20 + rh * ROWS.length); }
      ctx.fillStyle = m.key === model ? C.edge : C.mid;
      const t = m.short; ctx.fillText(t, x + (cw - ctx.measureText(t).width) / 2, tb.y + 14);
    });
    ROWS.forEach((rw, i) => {
      const y = tb.y + 20 + i * rh;
      ctx.fillStyle = C.ink; ctx.font = `${cw < 64 ? 10 : 11}px Pretendard Variable, sans-serif`;
      ctx.fillText(rw.label, tb.x, y + rh / 2 + 4);
      const vals = MODELS.map((m) => res.rows[m.key][rw.key]);
      const mx = Math.max(...vals.map(Math.abs), 1e-9);
      MODELS.forEach((m, j) => {
        const x = tb.x + lw + j * cw + 4, bwid = cw - 8, v = vals[j];
        ctx.fillStyle = m.key === model ? alpha(C.edge, 0.55) : alpha(C.ink2, 0.3);
        ctx.fillRect(x, y + 3, Math.max(1, (Math.max(v, 0) / mx) * bwid), rh - 6);
        ctx.fillStyle = C.ink; ctx.font = `${cw < 64 ? 10 : 11}px JetBrains Mono, monospace`;
        const t = rw.fmt(v); ctx.fillText(t, x + 3, y + rh / 2 + 4);
      });
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(tb.x, y + rh); ctx.lineTo(tb.x + tb.w, y + rh); ctx.stroke();
    });
    void H;
  }

  function update(recompute: boolean) {
    if (recompute) { world = makeWorld(seed); res = evaluate(world, n); }
    const r = res.rows[model];
    ro.set("pr", `P ${r.prec.toFixed(2)} · R ${r.rec.toFixed(2)} · 거리 비 ${r.ratio.toFixed(2)}`, r.rec < 0.85 || r.prec < 0.85 || r.ratio < 0.6 ? "hot" : "");
    ro.set("fid", `FID 원 좌표 ${r.fidRaw.toFixed(2)} · ReLU ${r.fidRelu.toFixed(1)}`);
    ro.set("mmd", `MMD² h 중앙값 ${r.mmdMed.toFixed(4)} · h 0.5 ${r.mmd05.toFixed(4)}`);
    draw();
  }

  const pm = presets(MODELS.map((m) => ({ key: m.key, label: m.label })), (k) => { model = k; update(false); }, model);
  pm.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 가로로 넘긴다
  let timer = 0;
  const pn = param({ label: "n", min: 50, max: 500, step: 50, value: n, fmt: (v) => String(v), aria: "생성 표본과 시험 표본의 수", onInput: (v) => { n = v; clearTimeout(timer); timer = window.setTimeout(() => update(true), 60); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 표본";
  reseed.addEventListener("click", () => { seed++; update(true); });
  tools.append(reseed);
  const ro = readouts();
  f.panel.append(pm.row, grid, tools, ro.row);
  f.onResize(draw);
  update(false);
}
