// 분할 단위와 예측 지평 (data/widgets.toml: split-unit-forecast).
// 1.3장 "범위의 명시"의 가상 배터리 셀 실험과 같은 생성 규칙: 셀 20개 × SOH 측정 40번(사이클 0–1000),
// 셀마다 열화 속도 fade ~ U(0.5e-4, 2.5e-4)/사이클, 초기 용량 cap0 ~ N(2.0, 0.05) Ah, SOH = 1 − fade·사이클 + N(0, 0.005).
// 입력은 (사이클, 초기 용량), 출력은 그 사이클의 SOH다. 새 셀이라면 "사이클 0에서 초기 용량만 보고 h 사이클 뒤의 SOH를 내다보는" 예측이다.
// 모델은 끝까지 키운 회귀 트리 하나다(본문 random forest를 이루는 나무 하나, 브라우저에서 바로 계산한다).
// 무작위 분할(측정의 25%)과 셀 단위 분할(셀 5개를 통째로)을 각각 20번 뽑아, 예측 지평(사이클)마다 테스트 MAE를 평균한다.
// 무작위 분할에서는 초기 용량이 셀의 지문이 되어 트리가 같은 셀의 앞뒤 측정으로 답하므로, 지평이 길어도 오차가 잡음 수준에 가깝게 머문다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

export const N_CELLS = 20, N_MEAS = 40, SPLITS = 20, NOISE = 0.005;
export const CYC = Array.from({ length: N_MEAS }, (_, k) => (1000 * k) / (N_MEAS - 1));
type Mode = "random" | "cell";

export function makeData(seed = 1303) {
  const { u, normal } = rng(seed);
  const fade = Array.from({ length: N_CELLS }, () => 0.5e-4 + 2e-4 * u());
  const cap0 = Array.from({ length: N_CELLS }, () => 2.0 + 0.05 * normal());
  const soh = fade.map((f) => CYC.map((c) => 1 - f * c + NOISE * normal()));
  // 분할 20쌍: 무작위는 측정마다 25% 확률로 테스트, 셀 단위는 셀 5개를 통째로 테스트
  const splits: Record<Mode, Uint8Array[]> = { random: [], cell: [] };
  for (let s = 0; s < SPLITS; s++) {
    const r = new Uint8Array(N_CELLS * N_MEAS);
    for (let i = 0; i < r.length; i++) r[i] = u() < 0.25 ? 1 : 0;
    const perm = Array.from({ length: N_CELLS }, (_, i) => i);
    for (let i = N_CELLS - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const held = new Set(perm.slice(0, 5));
    const c = new Uint8Array(N_CELLS * N_MEAS);
    for (let i = 0; i < c.length; i++) c[i] = held.has(Math.floor(i / N_MEAS)) ? 1 : 0;
    splits.random.push(r); splits.cell.push(c);
  }
  return { fade, cap0, soh, splits };
}
export type Data = ReturnType<typeof makeData>;

/** 회귀 트리(CART, 제곱오차, 끝까지 키움). 본문 random forest를 이루는 나무 하나와 같은 규칙이다(bootstrap 없음). */
type TreeNode = { leaf: true; v: number } | { leaf: false; f: number; t: number; l: TreeNode; r: TreeNode };
export function fitTree(F: Float64Array[], y: Float64Array, ids: number[]): TreeNode {
  const n = ids.length;
  let s = 0, s2 = 0;
  for (const i of ids) { s += y[i]; s2 += y[i] * y[i]; }
  const mean = s / n;
  if (n < 2 || s2 / n - mean * mean <= 2.220446049250313e-16) return { leaf: true, v: mean };
  let best = -Infinity, bf = -1, bt = 0;
  for (let f = 0; f < F.length; f++) {
    const x = F[f];
    const o = [...ids].sort((a, b) => x[a] - x[b]);
    let sl = 0;
    for (let i = 0; i < n - 1; i++) {
      sl += y[o[i]];
      if (x[o[i + 1]] <= x[o[i]] + 1e-7) continue;
      const nl = i + 1, sr = s - sl, score = (sl * sl) / nl + (sr * sr) / (n - nl);
      if (score > best) {
        best = score; bf = f;
        bt = (x[o[i]] + x[o[i + 1]]) / 2;
        if (bt === x[o[i + 1]]) bt = x[o[i]];
      }
    }
  }
  if (bf < 0) return { leaf: true, v: mean };
  const L = ids.filter((i) => F[bf][i] <= bt), R = ids.filter((i) => F[bf][i] > bt);
  return { leaf: false, f: bf, t: bt, l: fitTree(F, y, L), r: fitTree(F, y, R) };
}
const predictTree = (nd: TreeNode, x: number[]): number => (nd.leaf === true ? nd.v : predictTree(x[nd.f] <= nd.t ? nd.l : nd.r, x));

/** 한 분할에서 학습 행으로 트리를 키우고 테스트 행을 예측한다. 행 번호 = 셀 × 40 + 측정 번호 */
export function predictSplit(d: Data, test: Uint8Array) {
  const n = N_CELLS * N_MEAS;
  const f1 = new Float64Array(n), f2 = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { const c = Math.floor(i / N_MEAS), k = i % N_MEAS; f1[i] = CYC[k]; f2[i] = d.cap0[c]; y[i] = d.soh[c][k]; }
  const tr: number[] = [];
  for (let i = 0; i < n; i++) if (!test[i]) tr.push(i);
  const tree = fitTree([f1, f2], y, tr);
  const pred = new Map<number, number>();
  for (let i = 0; i < n; i++) if (test[i]) pred.set(i, predictTree(tree, [f1[i], f2[i]]));
  return pred;
}

/** 분할 20번에 걸쳐, 측정 번호(= 예측 지평)마다 평균 절대오차와 전체 평균 절대오차 */
export function maeCurves(d: Data) {
  const out = {} as Record<Mode, { curve: number[]; overall: number; first: Map<number, number> }>;
  for (const mode of ["random", "cell"] as Mode[]) {
    const sum = new Float64Array(N_MEAS), cnt = new Float64Array(N_MEAS);
    let first = new Map<number, number>();
    d.splits[mode].forEach((test, s) => {
      const pred = predictSplit(d, test);
      if (s === 0) first = pred;
      pred.forEach((p, i) => { const c = Math.floor(i / N_MEAS), k = i % N_MEAS; sum[k] += Math.abs(p - d.soh[c][k]); cnt[k]++; });
    });
    const tot = sum.reduce((x, v) => x + v, 0), n = cnt.reduce((x, v) => x + v, 0);
    out[mode] = { curve: Array.from(sum, (v, k) => v / cnt[k]), overall: tot / n, first };
  }
  return out;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "분할 단위와 예측 지평", "compare", narrow ? 1.25 : 0.62, "가상 배터리 셀 20개의 SOH 궤적과 학습·테스트 측정, 그리고 예측 지평별 테스트 MAE 곡선");
  const data = makeData();
  const res = maeCurves(data);
  let mode: Mode = "random", h = 31;
  const floor = NOISE * Math.sqrt(2 / Math.PI);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const L = narrow ? 44 : 54, R = narrow ? 14 : 70, top = 26, gap = 34;
    const h1 = (H - top - gap - 26) * 0.56, h2 = H - top - gap - 26 - h1;
    const y1 = top, y2 = top + h1 + gap;
    const X = (c: number) => L + (c / 1000) * (W - L - R);
    const test = data.splits[mode][0], pred = res[mode].first;
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.textBaseline = "alphabetic";

    // 위: SOH 궤적. 학습 측정은 흐린 점, 테스트 측정은 파란 점(분할 하나를 보인다)
    const S0 = 0.72, S1 = 1.015;
    const Y = (v: number) => y1 + h1 - ((v - S0) / (S1 - S0)) * h1;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(L, y1, W - L - R, h1);
    ctx.fillStyle = C.mid;
    ctx.fillText(narrow ? "SOH (가상 셀 20개)" : "SOH — 가상 셀 20개, 측정 40번", L, y1 - 9);
    ctx.font = "11px JetBrains Mono, monospace";
    for (const v of [0.75, 0.85, 0.95]) { ctx.fillText(v.toFixed(2), L - (narrow ? 34 : 38), Y(v) + 4); ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + 4, Y(v)); ctx.stroke(); }
    for (let c = 0; c < N_CELLS; c++) {
      const held = mode === "cell" && test[c * N_MEAS] === 1;
      ctx.strokeStyle = held ? C.ink2 : C.grid2; ctx.lineWidth = held ? 1.4 : 0.8;
      ctx.setLineDash(held ? [4, 3] : []);
      ctx.beginPath();
      CYC.forEach((cy, k) => (k ? ctx.lineTo(X(cy), Y(data.soh[c][k])) : ctx.moveTo(X(cy), Y(data.soh[c][k]))));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const pass of [0, 1]) for (let i = 0; i < N_CELLS * N_MEAS; i++) {
      if (test[i] !== pass) continue;
      const c = Math.floor(i / N_MEAS), k = i % N_MEAS;
      ctx.fillStyle = pass ? C.ink2 : C.grid2;
      ctx.beginPath(); ctx.arc(X(CYC[k]), Y(data.soh[c][k]), pass ? 2.6 : 1.6, 0, 7); ctx.fill();
    }
    // 지평 h의 테스트 측정: 실제(파란 점) → 예측(코랄 고리), 그 사이가 오차
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(CYC[h]), y1); ctx.lineTo(X(CYC[h]), y1 + h1); ctx.stroke();
    pred.forEach((p, i) => {
      if (i % N_MEAS !== h) return;
      const c = Math.floor(i / N_MEAS), x = X(CYC[h]);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, Y(data.soh[c][h])); ctx.lineTo(x, Y(p)); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, Y(p), 4, 0, 7); ctx.stroke();
    });

    // 아래: 예측 지평별 테스트 MAE(분할 20번 평균)
    const ymax = Math.max(...res.cell.curve, ...res.random.curve) * 1.12;
    const Y2 = (v: number) => y2 + h2 - (v / ymax) * h2;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(L, y2, W - L - R, h2);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid;
    ctx.fillText(narrow ? "테스트 MAE (분할 20번 평균)" : "테스트 MAE — 분할 20번 평균", L, y2 - 9);
    ctx.font = "11px JetBrains Mono, monospace";
    const step = ymax > 0.04 ? 0.02 : 0.01;
    for (let v = step; v < ymax; v += step) ctx.fillText(v.toFixed(2), L - (narrow ? 34 : 38), Y2(v) + 4);
    for (const c of [0, 500, 1000]) ctx.fillText(String(c), X(c) - (c ? (c === 1000 ? 26 : 10) : 0), y2 + h2 + 15);
    ctx.fillText("사이클", X(250) - 14, y2 + h2 + 15);
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(L, Y2(floor)); ctx.lineTo(W - R, Y2(floor)); ctx.stroke(); ctx.setLineDash([]);
    const names: Record<Mode, string> = { random: "무작위", cell: "셀 단위" };
    for (const m of (mode === "random" ? ["cell", "random"] : ["random", "cell"]) as Mode[]) {
      const on = m === mode, cv = res[m].curve;
      ctx.strokeStyle = on ? C.ink : C.mid; ctx.lineWidth = on ? 2.4 : 1.2;
      ctx.beginPath(); cv.forEach((v, k) => (k ? ctx.lineTo(X(CYC[k]), Y2(v)) : ctx.moveTo(X(CYC[k]), Y2(v)))); ctx.stroke();
      ctx.fillStyle = on ? C.edge : C.mid;
      ctx.beginPath(); ctx.arc(X(CYC[h]), Y2(cv[h]), on ? 5 : 3.5, 0, 7); ctx.fill();
    }
    if (!narrow) {
      // 오른쪽 끝 이름표: 서로 13px 안으로 붙으면 위아래로 벌린다
      const labs = [
        { t: names.cell, y: Y2(res.cell.curve[N_MEAS - 1]), col: mode === "cell" ? C.ink : C.mid },
        { t: names.random, y: Y2(res.random.curve[N_MEAS - 1]), col: mode === "random" ? C.ink : C.mid },
        { t: "잡음 하한", y: Y2(floor), col: C.mid },
      ].sort((p, q) => p.y - q.y);
      for (let i = 1; i < labs.length; i++) labs[i].y = Math.max(labs[i].y, labs[i - 1].y + 13);
      for (const lb of labs) { ctx.fillStyle = lb.col; ctx.fillText(lb.t, W - R + 6, Math.min(lb.y, y2 + h2 + 2) + 4); }
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(CYC[h]), y2); ctx.lineTo(X(CYC[h]), y2 + h2); ctx.stroke();
  }

  function update() {
    const a = res.random.curve[h], b = res.cell.curve[h];
    ro.set("random", `무작위 MAE ${a.toFixed(4)}`, mode === "random" ? "hot" : "");
    ro.set("cell", `셀 단위 MAE ${b.toFixed(4)}`, mode === "cell" ? "hot" : "");
    ro.set("ratio", `셀 단위 ÷ 무작위 ${(b / a).toFixed(1)}배`);
    ro.set("floor", `잡음 하한 ${floor.toFixed(4)}`);
    ro.set("all", `전 구간 평균 ${res.random.overall.toFixed(4)} · ${res.cell.overall.toFixed(4)}`);
    draw();
  }

  const modeBtns = presets<Mode>([{ key: "random", label: "무작위 분할" }, { key: "cell", label: "셀 단위 분할" }], (k) => { mode = k; update(); }, mode);
  modeBtns.row.classList.add("segmented");
  const ph = param({ label: "지평", min: 0, max: N_MEAS - 1, step: 1, value: h, fmt: (v) => `${Math.round(CYC[v])} 사이클`, aria: "예측 지평: 사이클 0에서 몇 사이클 뒤의 SOH를 예측하는가", onInput: (v) => { h = v; update(); } });
  ph.row.style.gridTemplateColumns = "2.4em minmax(0, 1fr) 6.4em";
  const grid = el("div", "param-grid");
  grid.append(ph.row);
  const ro = readouts();
  f.panel.append(modeBtns.row, grid, ro.row);
  f.onResize(draw);
  update();
}
