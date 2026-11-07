// 데이터 분할과 누수 (data/widgets.toml: split-leakage).
// 그룹(분자 골격·환자처럼 같은 출처의 샘플 묶음) 16개, 그룹마다 15개 샘플이 있다. 목표값 y = f(x) + u_g + ε에서
// u_g는 그룹마다 공유되는 값이다. k-최근접 이웃 회귀(k = 5)를 무작위 분할과 그룹 분할로 각각 학습·평가한다.
// 무작위 분할에서는 테스트 점의 이웃이 대개 같은 그룹의 학습 샘플이라 u_g를 베껴 와 테스트 오차가 작게 나온다.
// 그룹 분할은 처음 보는 그룹에 대한 일반화를 잰다. σ_g(그룹 효과의 크기)를 키우면 둘의 차이가 벌어진다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

const G = 16, PER = 15, N = G * PER, K = 5, TEST_GROUPS = 4, TEST_N = TEST_GROUPS * PER;
type Mode = "random" | "group";

interface Data { x1: Float64Array; x2: Float64Array; base: Float64Array; u: Float64Array; g: Int32Array; cx: number[]; cy: number[] }

function makeData(): Data {
  const r = rng(20251107);
  const cx: number[] = [], cy: number[] = [], u: number[] = [];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    cx.push(-1.8 + 1.2 * j + (r.u() - 0.5) * 0.5);
    cy.push(-1.8 + 1.2 * i + (r.u() - 0.5) * 0.5);
  }
  for (let k = 0; k < G; k++) u.push(r.normal());
  // u_g를 표준화해 σ_g가 그룹 효과의 표준편차가 되게 한다
  const mu = u.reduce((a, b) => a + b, 0) / G;
  const sd = Math.sqrt(u.reduce((a, b) => a + (b - mu) ** 2, 0) / G);
  const d: Data = { x1: new Float64Array(N), x2: new Float64Array(N), base: new Float64Array(N), u: new Float64Array(G), g: new Int32Array(N), cx, cy };
  for (let k = 0; k < G; k++) d.u[k] = (u[k] - mu) / sd;
  for (let n = 0; n < N; n++) {
    const k = Math.floor(n / PER);
    d.g[n] = k;
    d.x1[n] = cx[k] + 0.33 * r.normal();
    d.x2[n] = cy[k] + 0.33 * r.normal();
    d.base[n] = Math.sin(0.8 * d.x1[n]) + 0.3 * d.x2[n] + 0.15 * r.normal();
  }
  return d;
}

/** 시드로 테스트 집합을 고른다: 무작위 분할은 샘플 60개, 그룹 분할은 그룹 4개(= 샘플 60개). */
function split(mode: Mode, seed: number, d: Data): Uint8Array {
  const r = rng(seed * 104729 + (mode === "group" ? 7 : 3));
  const test = new Uint8Array(N);
  const perm = (m: number) => { const a = Array.from({ length: m }, (_, i) => i); for (let i = m - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  if (mode === "random") perm(N).slice(0, TEST_N).forEach((i) => (test[i] = 1));
  else { const gs = new Set(perm(G).slice(0, TEST_GROUPS)); for (let n = 0; n < N; n++) if (gs.has(d.g[n])) test[n] = 1; }
  return test;
}

/** k-NN 회귀로 테스트 점을 예측하고 RMSE와 '같은 그룹 이웃 비율'(테스트 점의 이웃 중 같은 그룹 학습 샘플의 비율)을 낸다. */
function evaluate(d: Data, y: Float64Array, test: Uint8Array) {
  const train: number[] = [], tst: number[] = [];
  for (let n = 0; n < N; n++) (test[n] ? tst : train).push(n);
  const pred = new Float64Array(N);
  let same = 0;
  const dist = new Float64Array(train.length);
  const idx = Array.from({ length: train.length }, (_, i) => i);
  for (const t of tst) {
    for (let a = 0; a < train.length; a++) { const s = train[a]; dist[a] = (d.x1[s] - d.x1[t]) ** 2 + (d.x2[s] - d.x2[t]) ** 2; }
    idx.sort((a, b) => dist[a] - dist[b]);
    let s = 0;
    for (let k = 0; k < K; k++) { const n = train[idx[k]]; s += y[n]; if (d.g[n] === d.g[t]) same++; }
    pred[t] = s / K;
  }
  let sse = 0;
  for (const t of tst) sse += (y[t] - pred[t]) ** 2;
  return { pred, rmse: Math.sqrt(sse / tst.length), same: same / (tst.length * K), tst };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "데이터 분할과 누수", "compare", narrow ? 1.4 : 0.56, "특성 공간의 그룹과 학습·테스트 샘플, 그리고 테스트 샘플의 실제값 대비 예측값");
  const data = makeData();
  let mode: Mode = "random", sg = 0.6, seed = 1;
  const y = new Float64Array(N);
  let tests: Record<Mode, Uint8Array>;
  let res: Record<Mode, ReturnType<typeof evaluate>>;

  function compute() {
    for (let n = 0; n < N; n++) y[n] = data.base[n] + sg * data.u[data.g[n]];
    tests = { random: split("random", seed, data), group: split("group", seed, data) };
    res = { random: evaluate(data, y, tests.random), group: evaluate(data, y, tests.group) };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const stacked = W < 460;
    const gap = 18, top = 24, pad = 10;
    const pw = stacked ? W - 2 * pad : (W - 2 * pad - gap) / 2;
    const ph = stacked ? (H - 2 * top - gap) / 2 : H - top - pad - 16;
    const side = Math.min(pw, ph);
    const boxes = stacked
      ? [{ x: (W - side) / 2, y: top }, { x: (W - side) / 2, y: top * 2 + side + gap - 8 }]
      : [{ x: pad + (pw - side) / 2, y: top }, { x: pad + pw + gap + (pw - side) / 2, y: top }];
    const test = tests[mode], r = res[mode];
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.textBaseline = "alphabetic";

    // 왼쪽: 특성 공간. 그룹은 원, 학습 샘플은 작은 점, 테스트 샘플은 강조색
    const A = boxes[0], R = 3.2;
    const P = (x: number, yy: number): [number, number] => [A.x + ((x + R) / (2 * R)) * side, A.y + ((R - yy) / (2 * R)) * side];
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(A.x, A.y, side, side);
    ctx.fillStyle = C.mid;
    ctx.fillText("특성 공간 (x₁, x₂)", A.x, A.y - 8);
    const rad = (0.62 / (2 * R)) * side;
    for (let k = 0; k < G; k++) {
      const [px, py] = P(data.cx[k], data.cy[k]);
      const held = mode === "group" && test[k * PER] === 1;
      ctx.strokeStyle = held ? C.edge : C.faint;
      ctx.lineWidth = held ? 1.6 : 1;
      ctx.setLineDash(held ? [4, 3] : []);
      ctx.beginPath(); ctx.arc(px, py, rad, 0, 7); ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const pass of [0, 1]) {
      for (let n = 0; n < N; n++) {
        if (test[n] !== pass) continue;
        const [px, py] = P(data.x1[n], data.x2[n]);
        ctx.fillStyle = pass ? C.edge : C.ink2;
        ctx.beginPath(); ctx.arc(px, py, pass ? 3.2 : 2.2, 0, 7); ctx.fill();
      }
    }

    // 오른쪽: 테스트 샘플의 실제값 대 예측값
    const B = boxes[1];
    let lo = Infinity, hi = -Infinity;
    for (let n = 0; n < N; n++) { lo = Math.min(lo, y[n]); hi = Math.max(hi, y[n]); }
    const m = (hi - lo) * 0.06; lo -= m; hi += m;
    const Q = (a: number, b: number): [number, number] => [B.x + ((a - lo) / (hi - lo)) * side, B.y + side - ((b - lo) / (hi - lo)) * side];
    ctx.strokeStyle = C.faint; ctx.strokeRect(B.x, B.y, side, side);
    ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(...Q(lo, lo)); ctx.lineTo(...Q(hi, hi)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.edge;
    for (const t of r.tst) { const [px, py] = Q(y[t], r.pred[t]); ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill(); }
    ctx.fillStyle = C.mid;
    ctx.fillText(`테스트: 실제 y → 예측 ŷ`, B.x, B.y - 8);
    ctx.font = "600 13px JetBrains Mono, monospace";
    ctx.fillStyle = C.ink;
    const lab = `RMSE ${r.rmse.toFixed(2)}`;
    ctx.fillText(lab, B.x + 8, B.y + 20);
  }

  function update() {
    compute();
    const a = res.random, b = res.group;
    ro.set("random", `무작위 분할 RMSE ${a.rmse.toFixed(2)}`, mode === "random" ? "hot" : "");
    ro.set("group", `그룹 분할 RMSE ${b.rmse.toFixed(2)}`, mode === "group" ? "hot" : "");
    ro.set("gap", `오차 비 ${(b.rmse / a.rmse).toFixed(1)}배`);
    ro.set("same", `같은 그룹 이웃 ${Math.round(res[mode].same * 100)}%`, res[mode].same > 0 ? "" : "ok");
    pageState.set("split", { mode, sigmaGroup: sg, seed, rmseRandom: a.rmse, rmseGroup: b.rmse }, root);
    draw();
  }

  const modeBtns = presets<Mode>([{ key: "random", label: "무작위 분할" }, { key: "group", label: "그룹 분할" }], (k) => { mode = k; update(); }, mode);
  modeBtns.row.classList.add("segmented");
  const ps = param({ label: "σ_g", min: 0, max: 1.5, step: 0.05, value: sg, aria: "그룹 효과의 표준편차 σ_g", onInput: (v) => { sg = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 분할";
  reseed.addEventListener("click", () => { seed++; update(); });
  tools.append(reseed);
  const ro = readouts();
  f.panel.append(modeBtns.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
