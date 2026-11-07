// Calibration과 reliability diagram (data/widgets.toml: calibration-diagram).
// 이진 분류 예측 N개를 고정 시드로 만든다: 참 로짓 z ~ N(0, 1.5²), 참 확률 σ(z), 레이블 y ~ Bernoulli(σ(z)).
// 모델은 로짓을 a배로 내놓는다: a = 1 잘 보정됨, a > 1 과신, a < 1 과소신뢰. 온도 T로 나누면(temperature scaling) q = σ(a·z / T).
// 같은 폭의 구간 M개로 나눠 구간마다 평균 예측 확률과 관측 빈도를 비교하고, ECE = Σ_b (n_b / N)·|관측 빈도_b − 평균 예측_b|를 계산한다.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const N = 5000;
const sigmoid = (t: number) => 1 / (1 + Math.exp(-t));
/** 고정 시드 표본: 참 로짓과 레이블 */
export const SAMPLE = (() => {
  const r = rng(314159);
  const z = new Float64Array(N), y = new Uint8Array(N);
  for (let i = 0; i < N; i++) { z[i] = 1.5 * r.normal(); y[i] = r.u() < sigmoid(z[i]) ? 1 : 0; }
  return { z, y };
})();
export function predictions(a: number, T: number) { return Array.from(SAMPLE.z, (z) => sigmoid((a * z) / T)); }

/** 구간별 통계와 ECE·Brier·NLL·정확도 */
export function evaluate(q: number[], y: ArrayLike<number>, M: number) {
  const cnt = new Array(M).fill(0), sq = new Array(M).fill(0), sy = new Array(M).fill(0);
  let brier = 0, nll = 0, acc = 0;
  for (let i = 0; i < q.length; i++) {
    const b = Math.min(M - 1, Math.floor(q[i] * M));
    cnt[b]++; sq[b] += q[i]; sy[b] += y[i];
    brier += (q[i] - y[i]) ** 2;
    const p = Math.min(1 - 1e-12, Math.max(1e-12, q[i]));
    nll -= y[i] ? Math.log(p) : Math.log(1 - p);
    acc += (q[i] >= 0.5 ? 1 : 0) === y[i] ? 1 : 0;
  }
  const n = q.length;
  const bins = cnt.map((c, b) => ({ count: c, conf: c ? sq[b] / c : NaN, freq: c ? sy[b] / c : NaN }));
  const ece = bins.reduce((s, b) => s + (b.count ? (b.count / n) * Math.abs(b.freq - b.conf) : 0), 0);
  return { bins, ece, brier: brier / n, nll: nll / n, acc: acc / n };
}

const MODELS = {
  calibrated: { label: "잘 보정됨", a: 1 },
  over: { label: "과신", a: 2.5 },
  under: { label: "과소신뢰", a: 0.4 },
} as const;
type MK = keyof typeof MODELS;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "Calibration과 reliability diagram", "manipulate", 0.72, "구간별 평균 예측 확률과 관측 빈도를 비교한 reliability diagram과 예측 확률의 분포");
  let model: MK = (attrs.model in MODELS ? attrs.model : "over") as MK;
  let lt = 0, M = 10; // T = 2^lt
  let ev = evaluate([], [], M);

  const PAD_L = 34, PAD_R = 12;
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const top = 12, histH = Math.round(H * 0.16), base = H - 30 - histH - 12, hb = H - 24;
    const X = (v: number) => PAD_L + v * (W - PAD_L - PAD_R);
    const Y = (v: number) => base - v * (base - top);
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (let k = 0; k <= 4; k++) {
      const v = k / 4;
      ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(1), Y(v)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(v), top); ctx.lineTo(X(v), base); ctx.stroke();
      ctx.fillText(v.toFixed(2).replace(/0$/, ""), 2, Y(v) + 4);
      const s = v.toFixed(2).replace(/0$/, "");
      ctx.fillText(s, Math.min(W - PAD_R - ctx.measureText(s).width, Math.max(PAD_L, X(v) - ctx.measureText(s).width / 2)), hb + 14);
    }
    // 대각선: 완벽한 보정
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
    // 구간 막대: 관측 빈도, 그리고 평균 예측과의 차이
    const bw = (W - PAD_L - PAD_R) / M;
    ev.bins.forEach((b, i) => {
      if (!b.count) return;
      const x0 = X(i / M) + 1.5, w = bw - 3;
      ctx.fillStyle = C.fill;
      ctx.fillRect(x0, Y(b.freq), w, base - Y(b.freq));
      ctx.fillStyle = alpha(C.edge, 0.3);
      const y1 = Y(Math.max(b.freq, b.conf)), y2 = Y(Math.min(b.freq, b.conf));
      ctx.fillRect(x0, y1, w, y2 - y1);
    });
    // 구간별 (평균 예측, 관측 빈도)
    const used = ev.bins.filter((b) => b.count);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); used.forEach((b, i) => (i ? ctx.lineTo(X(b.conf), Y(b.freq)) : ctx.moveTo(X(b.conf), Y(b.freq)))); ctx.stroke();
    ctx.fillStyle = C.edge;
    for (const b of used) { ctx.beginPath(); ctx.arc(X(b.conf), Y(b.freq), 3.5, 0, 7); ctx.fill(); }
    ctx.fillStyle = C.mid;
    ctx.fillText("관측 빈도", PAD_L + 4, top + 12);
    // 아래: 예측 확률 분포(sharpness)
    const maxc = Math.max(...ev.bins.map((b) => b.count), 1);
    ctx.fillStyle = alpha(C.mid, 0.45);
    ev.bins.forEach((b, i) => { const h = (b.count / maxc) * (histH - 16); ctx.fillRect(X(i / M) + 1.5, hb - h, bw - 3, h); });
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(X(0), hb); ctx.lineTo(X(1), hb); ctx.stroke();
    ctx.fillStyle = C.mid;
    ctx.fillText("예측 확률 분포", PAD_L + 4, hb - histH + 8);
  }

  function update() {
    const T = Math.pow(2, lt);
    ev = evaluate(predictions(MODELS[model].a, T), SAMPLE.y, M);
    ro.set("ece", `ECE = ${ev.ece.toFixed(3)}`, "hot");
    ro.set("brier", `Brier = ${ev.brier.toFixed(4)}`);
    ro.set("nll", `NLL = ${ev.nll.toFixed(4)}`);
    ro.set("acc", `정확도 = ${ev.acc.toFixed(3)}`);
    pageState.set("calibration", { model, temperature: T, bins: M }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const pre = presets((Object.keys(MODELS) as MK[]).map((k) => ({ key: k, label: MODELS[k].label })), (k) => { model = k; update(); }, model);
  const pm = param({ label: "M", min: 3, max: 20, step: 1, value: M, fmt: (v) => String(v), aria: "구간 수 M", onInput: (v) => { M = v; update(); } });
  const pt = param({ label: "T", min: -2, max: 2, step: 0.05, value: lt, fmt: (v) => Math.pow(2, v).toFixed(2), aria: "온도 T (temperature scaling, 로그 눈금)", onInput: (v) => { lt = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row, pm.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
  draw();
}
