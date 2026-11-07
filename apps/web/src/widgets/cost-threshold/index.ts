// 비용 비율과 임계값 (data/widgets.toml: cost-threshold).
// 21.6장 "보정이 전제다"의 합성 설비: 이상 신호 s ~ N(0, 1), 참 로짓 −3.5 + 1.5 s(고장률 약 7%), 보정된 확률 p = σ(로짓).
// 위: threshold마다 1000대당 기대 비용 E[c_FP·(정상인데 경보) + c_FN·(고장인데 놓침)]. 아래: 정상·고장 사례의 p 분포.
// 비용은 표본이 아니라 s에 대한 수치 적분으로 정확히 계산한다(본문 표본 추정 1184.8·522.5의 모집단 값 1203·535).
// 보류를 켜면 Chow 규칙을 비대칭 비용으로 넓힌 보류 구간 d/c_FN < p < 1 − d/c_FP에서 검토 비용 d를 낸다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

const A0 = -3.5, A1 = 1.5;                                  // 참 로짓 = A0 + A1·s
const NS = 1601, S0 = -8, S1 = 8, DS = (S1 - S0) / (NS - 1);
const sig = (z: number) => 1 / (1 + Math.exp(-z));
const S = Array.from({ length: NS }, (_, i) => S0 + i * DS);
const W = S.map((s) => (Math.exp(-0.5 * s * s) / Math.sqrt(2 * Math.PI)) * DS); // φ(s) ds
const P = S.map((s) => sig(A0 + A1 * s));
export const FAULT_RATE = P.reduce((a, p, i) => a + p * W[i], 0);

/** threshold thr에서 1000대당 기대 비용과 미탐지율(고장 중). reject가 있으면 그 구간에서는 검토 비용 d. */
export function evaluate(thr: number, cFN: number, cFP = 1, d: number | null = null) {
  const lo = d === null ? 2 : d / cFN, hi = d === null ? -1 : 1 - d / cFP;
  let cost = 0, miss = 0, rej = 0, err = 0, alarms = 0;
  for (let i = 0; i < NS; i++) {
    const p = P[i], w = W[i];
    if (p > lo && p < hi) { cost += d! * w; rej += w; continue; }
    const alarm = p > thr;
    if (alarm) { cost += cFP * (1 - p) * w; err += (1 - p) * w; alarms += w; }
    else { cost += cFN * p * w; err += p * w; miss += p * w; }
  }
  const cov = 1 - rej;
  return { cost: 1000 * cost, miss: miss / FAULT_RATE, reject: rej, coverage: cov, risk: cov > 0 ? err / cov : 0, alarmRate: alarms, lo, hi };
}

const U0 = Math.log10(0.003), U1 = 0;                        // 가로축: log10 p
type Pre = "even" | "text" | "lab";
const PRE: Record<Pre, number> = { even: 1, text: 20, lab: 25 };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "비용 비율과 임계값", "manipulate", narrow ? 0.98 : 0.58, "threshold에 따른 1000대당 기대 비용 곡선과, 정상·고장 사례의 보정된 고장 확률 분포");
  let ratio = 20, cand = 0.5, reject = false, d = 0.6;

  // 아래 그림의 밀도: u = log10 p에서 정상 (1−p)·g(u), 고장 p·g(u). g(u) = φ(s)·ln10 / (A1 (1−p))
  const dens = P.map((p, i) => (W[i] / DS) * Math.LN10 / (A1 * (1 - p)));
  const U = P.map((p) => Math.log10(p));
  const curveU = Array.from({ length: 181 }, (_, i) => U0 + ((U1 - U0) * i) / 180);

  function draw() {
    const { ctx, width: Wd, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, Wd, H);
    const L = narrow ? 40 : 50, R = 14, top = 22, split = H * 0.52, bottom = H - 26;
    const X = (u: number) => L + ((u - U0) / (U1 - U0)) * (Wd - L - R);
    const thrOpt = 1 / (1 + ratio);
    const costs = curveU.map((u) => evaluate(10 ** u, ratio).cost);
    const best = evaluate(thrOpt, ratio);
    const hiC = Math.max(...costs) * 1.05;
    const Y = (c: number) => split - 18 - (c / hiC) * (split - 18 - top);
    // 축과 눈금(위·아래 공통)
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textBaseline = "alphabetic";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, split - 18); ctx.lineTo(Wd - R, split - 18); ctx.moveTo(L, bottom); ctx.lineTo(Wd - R, bottom); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (const t of narrow ? [0.01, 0.1, 1] : [0.003, 0.01, 0.03, 0.1, 0.3, 1]) {
      const x = X(Math.log10(t));
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(x, bottom); ctx.lineTo(x, bottom + 4); ctx.stroke();
      const s = String(t); ctx.fillText(s, Math.min(x - ctx.measureText(s).width / 2, Wd - R - ctx.measureText(s).width), bottom + 16);
    }
    const step = [100, 200, 500, 1000, 2000, 5000].find((s) => hiC / s <= 3.2) ?? 10000;
    for (let c = 0; c <= hiC; c += step) {
      ctx.fillText(String(c), 4, Y(c) + 4);
      ctx.strokeStyle = C.faint; ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(L, Y(c)); ctx.lineTo(Wd - R, Y(c)); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("1000대당 기대 비용", L + 4, top - 6);
    // 보류 구간
    const r = reject ? evaluate(thrOpt, ratio, 1, d) : null;
    if (r && r.lo < r.hi) {
      ctx.fillStyle = alpha(C.good, 0.14);
      const x0 = X(Math.max(U0, Math.log10(r.lo))), x1 = X(Math.min(U1, Math.log10(r.hi)));
      ctx.fillRect(x0, top, x1 - x0, bottom - top);
      ctx.fillStyle = C.good; ctx.font = "600 11px Pretendard Variable, sans-serif";
      ctx.fillText("보류", x0 + 4, top + 26);
      ctx.setLineDash([6, 4]); ctx.strokeStyle = C.good; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(L, Y(r.cost)); ctx.lineTo(Wd - R, Y(r.cost)); ctx.stroke(); ctx.setLineDash([]);
    }
    // 비용 곡선
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); curveU.forEach((u, i) => (i ? ctx.lineTo(X(u), Y(costs[i])) : ctx.moveTo(X(u), Y(costs[i])))); ctx.stroke();
    // 최적 threshold(코랄)
    const xo = X(Math.log10(thrOpt));
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(xo, top); ctx.lineTo(xo, bottom); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(xo, Y(best.cost), 5, 0, 7); ctx.fill();
    // 아래: 정상·고장 사례 각각의 p 분포(집단마다 자기 최댓값으로 맞춘다). 칠한 넓이 = 오경보율(정상 중)·미탐지율(고장 중)
    const inRange = (i: number) => U[i] >= U0 && U[i] <= U1;
    const peakOf = (sel: (p: number) => number) => Math.max(...P.map((p, i) => (inRange(i) ? sel(p) * dens[i] : 0)));
    const normal = (p: number) => 1 - p, fault = (p: number) => p;
    const pk = { n: peakOf(normal), f: peakOf(fault) };
    const Y2 = (v: number, peak: number) => bottom - (v / peak) * (bottom - split - 16);
    const xc = X(Math.log10(cand)), lc = Math.log10(cand);
    const trace = (sel: (p: number) => number, peak: number, from: number, to: number, fill: string | null, stroke: string | null, w = 1.5) => {
      ctx.beginPath(); let started = false, lastX = 0;
      for (let i = 0; i < NS; i++) {
        if (!inRange(i) || U[i] < from || U[i] > to) continue;
        const x = X(U[i]), y = Y2(sel(P[i]) * dens[i], peak);
        if (!started) { if (fill) { ctx.moveTo(x, bottom); ctx.lineTo(x, y); } else ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        lastX = x;
      }
      if (!started) return;
      if (fill) { ctx.lineTo(lastX, bottom); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w; ctx.stroke(); }
    };
    trace(normal, pk.n, U0, U1, alpha(C.mid, 0.14), null);
    trace(normal, pk.n, lc, U1, alpha(C.mid, 0.4), null);      // 오경보: 정상인데 t보다 크다
    trace(fault, pk.f, U0, lc, alpha(C.ink2, 0.45), null);     // 미탐지: 고장인데 t보다 작다
    trace(normal, pk.n, U0, U1, null, C.mid, 1.3);
    trace(fault, pk.f, U0, U1, null, C.ink2, 2.2);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid; ctx.fillRect(L + 4, split, 9, 9); ctx.fillText("정상", L + 17, split + 9);
    ctx.fillStyle = C.ink2; ctx.fillRect(L + 52, split, 9, 9); ctx.fillText("고장", L + 65, split + 9);
    ctx.fillStyle = C.mid; ctx.fillText("사례의 p 분포", L + 98, split + 9);
    // 후보 threshold(끄는 점선)
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(xc, top); ctx.lineTo(xc, bottom); ctx.stroke(); ctx.setLineDash([]);
    const cc = evaluate(cand, ratio).cost;
    ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(xc, Y(cc), 4, 0, 7); ctx.fill();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const lab = `t ${cand < 0.1 ? cand.toFixed(3) : cand.toFixed(2)}`, lw = ctx.measureText(lab).width;
    ctx.fillText(lab, xc + 6 + lw > Wd - R ? xc - 6 - lw : xc + 6, top + 10);
  }

  function update() {
    const thrOpt = 1 / (1 + ratio);
    const best = evaluate(thrOpt, ratio), c = evaluate(cand, ratio);
    ro.set("opt", `t* = 1/(1+${ratio}) = ${thrOpt.toFixed(3)} · 비용 ${best.cost.toFixed(0)} · 미탐지 ${(best.miss * 100).toFixed(0)}%`, "hot");
    ro.set("cand", `t ${cand.toFixed(3)} · 비용 ${c.cost.toFixed(0)} · 미탐지 ${(c.miss * 100).toFixed(0)}%`, Math.abs(c.cost - best.cost) < 0.005 * best.cost ? "ok" : "");
    if (reject) {
      const r = evaluate(thrOpt, ratio, 1, d);
      ro.set("rej", r.lo < r.hi ? `보류 [${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}] · 비용 ${r.cost.toFixed(0)} · coverage ${r.coverage.toFixed(2)} · risk ${r.risk.toFixed(3)}` : "보류 구간 없음: 보류가 오류보다 비싸다", "ok");
    } else ro.set("rej", `보류 없음 · coverage 1.00 · risk ${best.risk.toFixed(3)}`);
    dRow.hidden = !reject;
    draw();
  }

  const pre = presets<Pre>([{ key: "even", label: "같은 비용 1:1" }, { key: "text", label: "본문 20:1" }, { key: "lab", label: "실습 25:1" }], (k) => { ratio = PRE[k]; pr.set(ratio); update(); }, "text");
  pre.row.classList.add("segmented");
  const pr = param({ label: "c_FN", min: 1, max: 100, step: 1, value: ratio, fmt: (v) => `${v}:1`, aria: "미탐지 비용 c_FN (오경보 비용 c_FP = 1)", onInput: (v) => { ratio = v; pre.select(v === 1 ? "even" : v === 20 ? "text" : v === 25 ? "lab" : null); update(); } });
  const pd = param({ label: "d", min: 0.05, max: 1, step: 0.05, value: d, aria: "보류(사람 검토) 비용 d, 오경보 비용 대비", onInput: (v) => { d = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pr.row);
  const dRow = el("div", "param-grid");
  dRow.append(pd.row);
  const tools = el("div");
  tools.style.cssText = "display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px;";
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 보류 허용`;
  sw.querySelector("input")!.addEventListener("change", (e) => { reject = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  tools.prepend(pre.row);
  f.panel.append(tools, grid, dRow, ro.row);

  // 그림 위에서 후보 threshold t를 끈다
  let dragging = false;
  const setFromEvent = (e: PointerEvent) => {
    const rc = f.canvas.getBoundingClientRect();
    const L = narrow ? 40 : 50, R = 14;
    const u = U0 + ((e.clientX - rc.left - L) / (f.width - L - R)) * (U1 - U0);
    const t = 10 ** Math.max(U0, Math.min(Math.log10(0.99), u));
    cand = Number(t.toPrecision(2));
    update();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); setFromEvent(e); });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) setFromEvent(e); });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.canvas.style.cursor = "ew-resize";
  f.canvas.tabIndex = 0;
  f.canvas.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    cand = Number(Math.min(0.99, Math.max(0.003, cand * (e.key === "ArrowRight" ? 1.15 : 1 / 1.15))).toPrecision(2));
    update();
  });
  f.onResize(draw);
  update();
}
