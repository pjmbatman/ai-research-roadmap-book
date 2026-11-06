// 손실함수가 고르는 대표값 (data/widgets.toml: loss-to-estimator).
// 위: 조건부분포 p(y|x)와 세 손실의 최적 예측(평균·중앙값·τ 분위수). 아래: 고른 손실의 기대값 E[ℓ(ŷ, y)]를 ŷ의 함수로 그린 곡선.
// 후보 예측 ŷ(점선)를 끌면 기대 손실이 어떻게 변하는지, 최소점이 어디인지 확인한다.
import { frame, param, presets, readouts, el, C, reducedMotion } from "../frame";
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
  let shown = NaN; // 화면에 그려진 최적 예측의 위치(손실을 바꾸면 새 위치로 미끄러진다)

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
    // 대표값: 고르지 않은 손실의 대표값은 흐린 선, 고른 손실의 최적 예측만 코랄(벤치마크 §3.4-15)
    const names: Record<Loss, string> = { mse: "평균", mae: "중앙값", pinball: `${Math.round(tau * 100)}% 분위수` };
    const at: Record<Loss, number> = { mse: stats.mean, mae: stats.median, pinball: stats.q };
    (["mse", "mae", "pinball"] as Loss[]).filter((l) => l !== loss).forEach((l) => {
      ctx.strokeStyle = "rgba(220,230,242,0.28)"; ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(X(at[l]), top + 10); ctx.lineTo(X(at[l]), split - 8); ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6;
    ctx.beginPath(); ctx.moveTo(X(shown), top + 4); ctx.lineTo(X(shown), split - 8); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.font = "600 12px Pretendard Variable, sans-serif";
    const label = `${names[loss]} ${at[loss].toFixed(2)}`;
    ctx.fillText(label, Math.min(X(shown) + 7, W - pad - ctx.measureText(label).width), top + 14);
    // 아래: 기대 손실 곡선
    const cx = xs.filter((_, i) => i % 3 === 0);
    const lo = Math.min(...curve), hi = Math.min(Math.max(...curve), lo + (loss === "mse" ? 12 : 4));
    const Y2 = (v: number) => bottom - ((Math.min(v, hi) - lo) / (hi - lo || 1)) * (bottom - split - 22);
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(pad, bottom); ctx.lineTo(W - pad, bottom); ctx.stroke();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); cx.forEach((x, i) => (i ? ctx.lineTo(X(x), Y2(curve[i])) : ctx.moveTo(X(x), Y2(curve[i])))); ctx.stroke();
    const opt = optimum();
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(X(opt), Y2(expected(loss, opt)), 5, 0, 7); ctx.fill();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("기대 손실", pad + 4, split + 24);
    // 후보 예측
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(cand), top); ctx.lineTo(X(cand), bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(X(cand), Y2(expected(loss, cand)), 4, 0, 7); ctx.fill();
    ctx.fillText("ŷ", X(cand) + 6, bottom - 6);
  }

  let anim = 0;
  function update(slide = false) {
    compute();
    const opt = optimum();
    ro.set("mean", `평균 ${stats.mean.toFixed(2)}`, loss === "mse" ? "hot" : "");
    ro.set("median", `중앙값 ${stats.median.toFixed(2)}`, loss === "mae" ? "hot" : "");
    ro.set("q", `${Math.round(tau * 100)}% 분위수 ${stats.q.toFixed(2)}`, loss === "pinball" ? "hot" : "");
    ro.set("cand", `ŷ ${cand.toFixed(2)} · 기대 손실 ${expected(loss, cand).toFixed(3)}`, Math.abs(cand - opt) < 0.05 ? "ok" : "");
    tauRow.hidden = loss !== "pinball";
    pageState.set("tau", tau, root);
    pageState.set("shape", shape, root);
    cancelAnimationFrame(anim);
    if (!slide || Number.isNaN(shown) || reducedMotion()) { shown = opt; draw(); return; }
    const from = shown, t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / 380), e = 1 - Math.pow(1 - k, 3);
      shown = from + (opt - from) * e;
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  const shapeBtns = presets((Object.keys(SHAPES) as Shape[]).map((k) => ({ key: k, label: SHAPES[k].label })), (k) => { shape = k; update(true); }, shape);
  const lossBtns = presets<Loss>([{ key: "mse", label: "제곱손실" }, { key: "mae", label: "절댓값손실" }, { key: "pinball", label: "pinball" }], (k) => { loss = k; update(true); }, loss);
  const pt = param({ label: "τ", min: 0.05, max: 0.95, step: 0.05, value: tau, aria: "pinball loss의 τ", onInput: (v) => { tau = v; update(); } });
  const tauRow = el("div", "param-grid");
  tauRow.append(pt.row);
  const ro = readouts();
  lossBtns.row.classList.add("segmented");
  f.panel.append(lossBtns.row, shapeBtns.row, tauRow, ro.row);

  // 그림 위에서 ŷ 끌기
  let dragging = false;
  const setFromEvent = (e: PointerEvent) => {
    const r = f.canvas.getBoundingClientRect();
    const x = X0 + ((e.clientX - r.left - 28) / (f.width - 56)) * (X1 - X0);
    cand = Math.round(Math.max(X0, Math.min(X1, x)) * 20) / 20;
    update();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); setFromEvent(e); });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) setFromEvent(e); });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.canvas.style.cursor = "ew-resize";
  f.onResize(draw);
  update();
}
