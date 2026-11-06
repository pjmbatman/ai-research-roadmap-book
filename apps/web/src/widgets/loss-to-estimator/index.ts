// 손실함수가 고르는 대표값 (data/widgets.toml: loss-to-estimator).
// 위: 조건부분포 p(y|x)와 세 손실의 최적 예측(평균·중앙값·τ 분위수). 아래: 고른 손실의 기대값 E[ℓ(ŷ, y)]를 ŷ의 함수로 그린 곡선.
// 후보 예측 ŷ(점선)를 끌면 기대 손실이 어떻게 변하는지, 최소점이 어디인지 확인한다.
import { frame, param, presets, readouts, el, C } from "../frame";
import { pageState } from "../../client/store";

const X0 = -3, X1 = 6, N = 721;
const xs = Array.from({ length: N }, (_, i) => X0 + ((X1 - X0) * i) / (N - 1));
const dx = (X1 - X0) / (N - 1);
const gauss = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
type Shape = "normal" | "skewed" | "bimodal";
const SHAPES: Record<Shape, { label: string; pdf: (x: number) => number }> = {
  normal: { label: "대칭(가우시안)", pdf: (x) => gauss(x, 1, 0.7) },
  skewed: { label: "비대칭(지수분포)", pdf: (x) => (x >= 0 ? Math.exp(-x) : 0) },
  bimodal: { label: "쌍봉", pdf: (x) => 0.5 * gauss(x, -1, 0.3) + 0.5 * gauss(x, 1, 0.3) },
};
type Loss = "mse" | "mae" | "pinball";

export function mount(root: HTMLElement) {
  const f = frame(root, "손실함수가 고르는 대표값", "manipulate", 0.66, "분포 곡선과 평균·중앙값·분위수 위치, 그리고 선택한 손실의 기대값 곡선");
  let shape: Shape = "skewed", tau = 0.9, loss: Loss = "pinball", cand = 2.0;
  let p: number[] = [];
  let curve: number[] = [];
  let stats = { mean: 0, median: 0, q: 0, mode: 0 };

  const quantile = (prob: number) => { let c = 0; for (let i = 0; i < N; i++) { c += p[i] * dx; if (c >= prob) return xs[i]; } return X1; };
  const ell = (l: Loss, yhat: number, y: number) => {
    const u = y - yhat;
    return l === "mse" ? u * u : l === "mae" ? Math.abs(u) : u * (tau - (u < 0 ? 1 : 0));
  };
  const expected = (l: Loss, yhat: number) => { let s = 0; for (let i = 0; i < N; i++) if (p[i] > 1e-9) s += ell(l, yhat, xs[i]) * p[i] * dx; return s; };

  function compute() {
    const raw = xs.map(SHAPES[shape].pdf);
    const z = raw.reduce((a, b) => a + b, 0) * dx;
    p = raw.map((v) => v / z);
    const mean = xs.reduce((s, x, i) => s + x * p[i] * dx, 0);
    let mi = 0; p.forEach((v, i) => { if (v > p[mi]) mi = i; });
    stats = { mean, median: quantile(0.5), q: quantile(tau), mode: xs[mi] };
    curve = xs.filter((_, i) => i % 3 === 0).map((x) => expected(loss, x));
  }

  const optimum = () => (loss === "mse" ? stats.mean : loss === "mae" ? stats.median : stats.q);
  const LCOL: Record<Loss, string> = { mse: C.ink, mae: C.blue, pinball: C.edge };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pad = 28, top = 16, split = H * 0.58, bottom = H - 24;
    const X = (x: number) => pad + ((x - X0) / (X1 - X0)) * (W - 2 * pad);
    const pmax = Math.max(...p) * 1.12;
    const Y = (v: number) => split - 8 - (v / pmax) * (split - 8 - top - 14);
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, split - 8); ctx.lineTo(W - pad, split - 8); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let k = Math.ceil(X0); k <= X1; k++) ctx.fillText(String(k), X(k) - 3, split + 6);
    // τ 아래 면적
    if (loss === "pinball") {
      ctx.fillStyle = C.edgeSoft;
      ctx.beginPath(); ctx.moveTo(X(X0), Y(0));
      xs.forEach((x, i) => { if (x <= stats.q) ctx.lineTo(X(x), Y(p[i])); });
      ctx.lineTo(X(stats.q), Y(0)); ctx.closePath(); ctx.fill();
    }
    // 밀도
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(p[i])) : ctx.moveTo(X(x), Y(p[i])))); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.fillText("p(y | x)", pad + 4, top + 4);
    // 대표값 표시
    const marks: [string, number, string][] = [["평균 · MSE", stats.mean, C.ink], ["중앙값 · MAE", stats.median, C.blue], [`${Math.round(tau * 100)}% 분위수 · pinball`, stats.q, C.edge]];
    marks.forEach(([label, x, col], k) => {
      ctx.strokeStyle = col; ctx.lineWidth = loss === (["mse", "mae", "pinball"] as Loss[])[k] ? 2.4 : 1.2;
      ctx.beginPath(); ctx.moveTo(X(x), top + 10); ctx.lineTo(X(x), split - 8); ctx.stroke();
      ctx.fillStyle = col; ctx.font = "600 11px JetBrains Mono, monospace";
      const tx = Math.min(X(x) + 5, W - pad - ctx.measureText(label).width);
      ctx.fillText(label, tx, top + 16 + k * 14);
    });
    // 아래: 기대 손실 곡선
    const cx = xs.filter((_, i) => i % 3 === 0);
    const lo = Math.min(...curve), hi = Math.min(Math.max(...curve), lo + (loss === "mse" ? 12 : 4));
    const Y2 = (v: number) => bottom - ((Math.min(v, hi) - lo) / (hi - lo || 1)) * (bottom - split - 22);
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(pad, bottom); ctx.lineTo(W - pad, bottom); ctx.stroke();
    ctx.strokeStyle = LCOL[loss]; ctx.lineWidth = 2;
    ctx.beginPath(); cx.forEach((x, i) => (i ? ctx.lineTo(X(x), Y2(curve[i])) : ctx.moveTo(X(x), Y2(curve[i])))); ctx.stroke();
    const opt = optimum();
    ctx.fillStyle = LCOL[loss];
    ctx.beginPath(); ctx.arc(X(opt), Y2(expected(loss, opt)), 5, 0, 7); ctx.fill();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(`E[ℓ(ŷ, y)] · ${loss === "mse" ? "제곱손실" : loss === "mae" ? "절댓값손실" : `pinball τ=${tau.toFixed(2)}`}`, pad + 4, split + 24);
    // 후보 예측
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(cand), top); ctx.lineTo(X(cand), bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(X(cand), Y2(expected(loss, cand)), 4, 0, 7); ctx.fill();
    ctx.fillText("ŷ", X(cand) + 6, bottom - 6);
  }

  function update() {
    compute();
    const opt = optimum();
    ro.set("mean", `평균 ${stats.mean.toFixed(2)}`, loss === "mse" ? "hot" : "");
    ro.set("median", `중앙값 ${stats.median.toFixed(2)}`, loss === "mae" ? "hot" : "");
    ro.set("q", `${Math.round(tau * 100)}% 분위수 ${stats.q.toFixed(2)}`, loss === "pinball" ? "hot" : "");
    ro.set("mode", `최빈값 ${stats.mode.toFixed(2)}`);
    ro.set("cand", `ŷ = ${cand.toFixed(2)} · 기대 손실 ${expected(loss, cand).toFixed(3)} (최소 ${expected(loss, opt).toFixed(3)})`, Math.abs(cand - opt) < 0.05 ? "ok" : "");
    pageState.set("tau", tau, root);
    pageState.set("shape", shape, root);
    draw();
  }

  const shapeBtns = presets((Object.keys(SHAPES) as Shape[]).map((k) => ({ key: k, label: SHAPES[k].label })), (k) => { shape = k; update(); }, shape);
  const lossBtns = presets<Loss>([{ key: "mse", label: "제곱손실" }, { key: "mae", label: "절댓값손실" }, { key: "pinball", label: "pinball" }], (k) => { loss = k; update(); }, loss);
  const pt = param({ label: "τ", min: 0.05, max: 0.95, step: 0.05, value: tau, aria: "pinball loss의 τ", onInput: (v) => { tau = v; update(); } });
  const pc = param({ label: "ŷ", min: X0, max: X1, step: 0.05, value: cand, aria: "후보 예측 ŷ", onInput: (v) => { cand = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row, pc.row);
  const ro = readouts();
  lossBtns.row.classList.add("segmented");
  f.panel.append(shapeBtns.row, lossBtns.row, grid, ro.row);

  // 그림 위에서 ŷ 끌기
  let dragging = false;
  const setFromEvent = (e: PointerEvent) => {
    const r = f.canvas.getBoundingClientRect();
    const x = X0 + ((e.clientX - r.left - 28) / (f.width - 56)) * (X1 - X0);
    cand = Math.round(Math.max(X0, Math.min(X1, x)) * 20) / 20;
    pc.set(cand);
    update();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); setFromEvent(e); });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) setFromEvent(e); });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.canvas.style.cursor = "ew-resize";
  f.onResize(draw);
  update();
}
