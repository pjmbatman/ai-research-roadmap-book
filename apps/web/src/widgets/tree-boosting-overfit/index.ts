// 트리 깊이와 boosting (data/widgets.toml: tree-boosting-overfit).
// 16.4장의 가상 합금 데이터(make_alloy와 같은 식, 이 위젯의 고정 시드로 학습 1000개·시험 1000개):
//   y = 200 + 3000·cu + 5000·mg + 40·[520 < T < 600 ∧ time > 3] + 15·sin(time) + 라인 효과 + N(0, 10²)  [MPa]
//   특징 7개: temp_c, time_h, cu, mg, line(0–4), noise_x(무관), lot_id(0–999, 무관)
// 제곱 손실 gradient boosting: F₀ = ȳ, 단계마다 잔차에 깊이 d의 회귀 트리(CART, 제곱오차 최소 분할, 문턱은 중간값)를 맞추고 F += ν·h.
// 깊이 d·학습률 ν를 바꾸면 처음부터 다시 학습하고, 단계 수 m은 이미 계산한 경로 위의 위치만 바꾼다.
// 아래 그림은 시효 온도 단면(time 6 h, cu 0.025, mg 0.01, line 0, noise_x 0, lot_id 500)의 참 평균과 모델 예측이다.
import { frame, param, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const NF = 7, NTR = 1000, NTE = 1000, MMAX = 400;
const LINE = [0, 10, -5, 15, 0];
const mean = (x: number[]) => 200 + 3000 * x[2] + 5000 * x[3] + (x[0] > 520 && x[0] < 600 && x[1] > 3 ? 40 : 0) + 15 * Math.sin(x[1]) + LINE[x[4]];
function makeAlloy(n: number, r: ReturnType<typeof rng>) {
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = [450 + 200 * r.u(), 0.5 + 11.5 * r.u(), 0.05 * r.u(), 0.02 * r.u(), Math.floor(5 * r.u()), r.normal(), Math.floor(1000 * r.u())];
    X.push(x); y.push(mean(x) + 10 * r.normal());
  }
  return { X, y };
}
export const DATA = (() => { const r = rng(1604); const tr = makeAlloy(NTR, r), te = makeAlloy(NTE, r); return { tr, te }; })();
const SLICE_T = Array.from({ length: 101 }, (_, i) => 450 + 2 * i);
const SLICE = SLICE_T.map((t) => [t, 6, 0.025, 0.01, 0, 0, 500]);
const SLICE_TRUE = SLICE.map(mean);

// 특징 우선(column-major) 배열과 특징별 정렬 순서는 한 번만 만든다
const XC = Array.from({ length: NF }, (_, j) => Float64Array.from(DATA.tr.X, (x) => x[j]));
const ORDER = XC.map((col) => Int32Array.from({ length: NTR }, (_, i) => i).sort((a, b) => col[a] - col[b]));

interface Tree { feat: number[]; thr: number[]; left: number[]; right: number[]; val: number[] }

/** 잔차 r에 깊이 depth 이하의 회귀 트리를 맞춘다(sklearn DecisionTreeRegressor와 같은 분할 규칙) */
export function fitTree(r: Float64Array, depth: number): Tree {
  const T: Tree = { feat: [], thr: [], left: [], right: [], val: [] };
  const goLeft = new Uint8Array(NTR);
  const build = (ord: Int32Array[], d: number): number => {
    const id = T.feat.length;
    const n = ord[0].length;
    let S = 0;
    for (let p = 0; p < n; p++) S += r[ord[0][p]];
    T.feat.push(-1); T.thr.push(0); T.left.push(-1); T.right.push(-1); T.val.push(S / n);
    if (d >= depth || n < 2) return id;
    let best = (S * S) / n + 1e-9 * Math.abs(S * S / n) + 1e-12, bj = -1, bt = 0;
    for (let j = 0; j < NF; j++) {
      const o = ord[j], col = XC[j];
      let sL = 0;
      for (let p = 0; p < n - 1; p++) {
        sL += r[o[p]];
        const a = col[o[p]], b = col[o[p + 1]];
        if (a === b) continue;
        const nL = p + 1, g = (sL * sL) / nL + ((S - sL) * (S - sL)) / (n - nL);
        if (g > best) { best = g; bj = j; bt = (a + b) / 2; }
      }
    }
    if (bj < 0) return id;
    const col = XC[bj];
    for (let p = 0; p < n; p++) { const i = ord[0][p]; goLeft[i] = col[i] <= bt ? 1 : 0; }
    const L: Int32Array[] = [], R: Int32Array[] = [];
    for (let j = 0; j < NF; j++) {
      const o = ord[j];
      let nl = 0;
      for (let p = 0; p < n; p++) nl += goLeft[o[p]];
      const l = new Int32Array(nl), rr = new Int32Array(n - nl);
      let a = 0, b = 0;
      for (let p = 0; p < n; p++) { const i = o[p]; if (goLeft[i]) l[a++] = i; else rr[b++] = i; }
      L.push(l); R.push(rr);
    }
    T.feat[id] = bj; T.thr[id] = bt;
    const li = build(L, d + 1);
    const ri = build(R, d + 1);
    T.left[id] = li; T.right[id] = ri;
    return id;
  };
  build(ORDER, 0);
  return T;
}
const predict = (T: Tree, x: number[]) => { let k = 0; while (T.feat[k] >= 0) k = x[T.feat[k]] <= T.thr[k] ? T.left[k] : T.right[k]; return T.val[k]; };

/** 깊이·학습률을 정하면 MMAX 단계까지의 학습·시험 RMSE와 단면 예측을 계산한다 */
export function boost(depth: number, lr: number) {
  const ytr = DATA.tr.y, yte = DATA.te.y;
  const y0 = ytr.reduce((s, v) => s + v, 0) / NTR;
  const F = new Float64Array(NTR).fill(y0), Ft = new Float64Array(NTE).fill(y0), Fs = new Float64Array(SLICE.length).fill(y0);
  const r = new Float64Array(NTR);
  const rm = (a: Float64Array, y: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / y.length);
  const train = [rm(F, ytr)], test = [rm(Ft, yte)], slice = [Float64Array.from(Fs)];
  let leaves = 0;
  for (let m = 1; m <= MMAX; m++) {
    for (let i = 0; i < NTR; i++) r[i] = ytr[i] - F[i];
    const T = fitTree(r, depth);
    if (m === 1) leaves = T.feat.filter((v) => v < 0).length;
    for (let i = 0; i < NTR; i++) F[i] += lr * predict(T, DATA.tr.X[i]);
    for (let i = 0; i < NTE; i++) Ft[i] += lr * predict(T, DATA.te.X[i]);
    for (let i = 0; i < SLICE.length; i++) Fs[i] += lr * predict(T, SLICE[i]);
    train.push(rm(F, ytr)); test.push(rm(Ft, yte)); slice.push(Float64Array.from(Fs));
  }
  return { train, test, slice, leaves };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "트리 깊이와 boosting", "manipulate", 0.62, "boosting 단계에 따른 학습·시험 RMSE 곡선과, 시효 온도 단면에서 참 평균과 모델 예측");
  let depth = 3, llr = -1, m = 100; // 학습률 = 10^llr
  let res = boost(depth, 10 ** llr);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 12, top = 18, split = Math.round(H * 0.52), bottom = H - 22;
    // 위: RMSE 곡선
    const X = (k: number) => PL + (k / MMAX) * (W - PL - PR);
    const ymax = 60;
    const Y = (v: number) => split - 18 - (Math.min(v, ymax) / ymax) * (split - 18 - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [0, 20, 40, 60]) { ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke(); ctx.fillStyle = C.mid; ctx.fillText(String(v), PL - 6 - ctx.measureText(String(v)).width, Y(v) + 4); }
    ctx.strokeStyle = C.good; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(PL, Y(10)); ctx.lineTo(W - PR, Y(10)); ctx.stroke(); ctx.setLineDash([]);
    const curve = (arr: number[], col: string, w: number) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); arr.forEach((v, k) => (k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v)))); ctx.stroke(); };
    curve(res.train, C.ink2, 1.8);
    curve(res.test, C.edge, 2.4);
    for (const k of [0, 100, 200, 300, 400]) ctx.fillText(String(k), X(k) - ctx.measureText(String(k)).width / 2, split - 4);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(m), top); ctx.lineTo(X(m), split - 18); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X(m), Y(res.test[m]), 4, 0, 7); ctx.fill();
    ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(X(m), Y(res.train[m]), 3.5, 0, 7); ctx.fill();
    // 범례
    let lx = PL + 4;
    for (const [nm, col] of [["시험", C.edge], ["학습", C.ink2], ["잡음 10 MPa", C.good]] as [string, string][]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(nm.startsWith("잡음") ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, top - 6); ctx.lineTo(lx + 14, top - 6); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, top - 2);
      lx += 30 + ctx.measureText(nm).width;
    }
    if (lx + ctx.measureText("RMSE [MPa]").width < W - PR) ctx.fillText("RMSE [MPa]", W - PR - ctx.measureText("RMSE [MPa]").width, top - 2);
    // 아래: 온도 단면
    const top2 = split + 14;
    const X2 = (t: number) => PL + ((t - 450) / 200) * (W - PL - PR);
    const lo = 280, hi = 400;
    const Y2 = (v: number) => bottom - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (bottom - top2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [300, 350, 400]) { ctx.beginPath(); ctx.moveTo(PL, Y2(v)); ctx.lineTo(W - PR, Y2(v)); ctx.stroke(); ctx.fillStyle = C.mid; ctx.fillText(String(v), PL - 6 - ctx.measureText(String(v)).width, Y2(v) + 4); }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); SLICE_TRUE.forEach((v, i) => (i ? ctx.lineTo(X2(SLICE_T[i]), Y2(v)) : ctx.moveTo(X2(SLICE_T[i]), Y2(v)))); ctx.stroke(); ctx.setLineDash([]);
    const s = res.slice[m];
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); s.forEach((v, i) => (i ? ctx.lineTo(X2(SLICE_T[i]), Y2(v)) : ctx.moveTo(X2(SLICE_T[i]), Y2(v)))); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (const t of [450, 500, 550, 600, 650]) ctx.fillText(String(t), X2(t) - ctx.measureText(String(t)).width / 2, bottom + 14);
    ctx.fillStyle = alpha(C.mid, 1);
    ctx.fillText("시효 온도 단면 [°C] · 점선 = 참 평균", PL + 4, top2 + 2);
  }

  let raf = 0, dirty = false;
  function update(retrain: boolean) {
    if (retrain) dirty = true;
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (dirty) { res = boost(depth, 10 ** llr); dirty = false; }
      const best = res.test.slice(1).reduce((b, v, i) => (v < res.test[b] ? i + 1 : b), 1);
      ro.set("now", `단계 ${m}: 학습 ${res.train[m].toFixed(1)} · 시험 ${res.test[m].toFixed(1)} MPa`);
      ro.set("gap", `격차 ${(res.test[m] - res.train[m]).toFixed(1)} MPa`, res.test[m] - res.train[m] > 8 ? "hot" : "");
      ro.set("best", `시험 최소 ${res.test[best].toFixed(1)} MPa (단계 ${best})`, "ok");
      ro.set("leaf", `트리 하나의 잎 ${res.leaves}개`);
      pageState.set("tree-boosting-overfit", { depth, lr: 10 ** llr, m, test: res.test[m] }, root);
      draw();
    });
  }
  const fmtLr = (v: number) => { const a = 10 ** v; return a >= 0.1 ? a.toFixed(2) : a.toFixed(3); };
  const pd = param({ label: "깊이", min: 1, max: 10, step: 1, value: depth, fmt: (v) => String(v), aria: "트리의 최대 깊이", onInput: (v) => { depth = v; update(true); } });
  const pl = param({ label: "ν", min: -2, max: 0, step: 0.05, value: llr, fmt: fmtLr, aria: "학습률 ν (로그 눈금)", onInput: (v) => { llr = v; update(true); } });
  const pm = param({ label: "m", min: 1, max: MMAX, step: 1, value: m, fmt: (v) => String(v), aria: "boosting 단계 수 m (더한 트리 수)", onInput: (v) => { m = v; update(false); } });
  const grid = el("div", "param-grid");
  grid.append(pd.row, pl.row, pm.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update(false);
}
