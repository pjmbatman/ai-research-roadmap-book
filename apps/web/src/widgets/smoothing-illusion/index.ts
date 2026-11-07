// 곡선 smoothing 착시 (data/widgets.toml: smoothing-illusion).
// 17.8장 "대시보드와 원본의 일치"의 합성 곡선: 기저 2·exp(−s/150) + 0.3 + 0.0008·max(s − 800, 0)에 잡음 0.08·N(0, 1),
// step 601–605에 +3.0 spike. 잡음은 위젯의 고정 시드(본문 셀의 NumPy 난수와 다르다)라 요약 숫자가 본문과 조금 다르다.
// 대시보드의 EMA: s_k = w·s_{k−1} + (1 − w)·x_k, s_0 = 0, 보정하면 s_k / (1 − w^k).
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";

export const N = 1000;
export const BASE = Float64Array.from({ length: N }, (_, i) => { const s = i + 1; return 2 * Math.exp(-s / 150) + 0.3 + 0.0008 * Math.max(s - 800, 0); });
export const LOSS = (() => {
  const r = rng(7987);   // 본문 셀의 요약 숫자에 가까운 잡음을 내는 시드
  const x = Float64Array.from(BASE, (b) => b + 0.08 * r.normal());
  for (let i = 600; i < 605; i++) x[i] += 3.0;
  return x;
})();

export function ema(x: Float64Array, w: number, debias: boolean) {
  const out = new Float64Array(x.length);
  let s = 0, wk = 1;
  for (let i = 0; i < x.length; i++) {
    s = w * s + (1 - w) * x[i];
    wk *= w;
    out[i] = debias && wk < 1 ? s / (1 - wk) : s;
  }
  return out;
}

/** 본문 셀과 같은 요약: 상승 구간(step 900–1000)의 평균 (기저 − 곡선), spike 근처(step 596–615)의 최댓값 */
export function summarize(sm: Float64Array) {
  let lag = 0, raw = 0;
  for (let i = 899; i < N; i++) { lag += BASE[i] - sm[i]; raw += BASE[i] - LOSS[i]; }
  let spikeSm = -Infinity, spikeRaw = -Infinity;
  for (let i = 595; i < 615; i++) { spikeSm = Math.max(spikeSm, sm[i]); spikeRaw = Math.max(spikeRaw, LOSS[i]); }
  return { lag: lag / (N - 899), lagRaw: raw / (N - 899), spikeSm, spikeRaw, first: sm[0], firstRaw: LOSS[0] };
}

const PRESETS = { w0: { label: "원본 0", w: 0 }, w6: { label: "0.6", w: 0.6 }, w9: { label: "0.9", w: 0.9 }, w99: { label: "0.99", w: 0.99 } } as const;
type PK = keyof typeof PRESETS;
const Z0 = 700, Z1 = 1000;

export function mount(root: HTMLElement) {
  const f = frame(root, "곡선 smoothing 착시", "manipulate", 0.56, "기록된 손실, 기저 곡선, EMA로 smoothing한 곡선과 step 700–1000 확대");
  let w = 0.6, debias = true;
  let sm = ema(LOSS, w, debias);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 10, top = 18, split = Math.round(H * 0.6), bottom = H - 22;
    const X = (i: number, a = 0, b = N) => PL + ((i - a) / (b - a)) * (W - PL - PR);
    // 위: 전체 0–4
    const y0 = split - 16, Y = (v: number) => y0 - (Math.min(Math.max(v, 0), 4) / 4) * (y0 - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [0, 1, 2, 3, 4]) {
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      ctx.fillStyle = C.mid; ctx.fillText(String(v), PL - 14, Y(v) + 4);
    }
    // 확대 구간 표시
    ctx.fillStyle = alpha(C.mid, 0.08);
    ctx.fillRect(X(Z0), top, X(Z1) - X(Z0), y0 - top);
    for (const s of [0, 500, 1000]) { const t = String(s), tw = ctx.measureText(t).width; ctx.fillStyle = C.mid; ctx.fillText(t, Math.min(W - PR - tw, Math.max(PL, X(s) - tw / 2)), y0 + 13); }
    const line = (arr: Float64Array, Xf: (i: number) => number, Yf: (v: number) => number, a: number, b: number) => {
      ctx.beginPath();
      for (let i = a; i < b; i++) i === a ? ctx.moveTo(Xf(i + 1), Yf(arr[i])) : ctx.lineTo(Xf(i + 1), Yf(arr[i]));
      ctx.stroke();
    };
    ctx.strokeStyle = alpha(C.mid, 0.55); ctx.lineWidth = 1;
    line(LOSS, (i) => X(i), Y, 0, N);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    line(BASE, (i) => X(i), Y, 0, N);
    ctx.setLineDash([]);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    line(sm, (i) => X(i), Y, 0, N);
    ctx.fillStyle = C.mid;
    ctx.fillText("손실", PL + 4, top - 5);
    const sp = "spike", spw = ctx.measureText(sp).width;
    ctx.fillText(sp, X(603) - spw / 2, top + 10);
    // 아래: step 700–1000 확대
    const lo = 0.15, hi = 0.75;
    const Yz = (v: number) => bottom - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (bottom - split - 6);
    const Xz = (i: number) => X(i, Z0, Z1);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [0.2, 0.4, 0.6]) {
      ctx.beginPath(); ctx.moveTo(PL, Yz(v)); ctx.lineTo(W - PR, Yz(v)); ctx.stroke();
      ctx.fillStyle = C.mid; ctx.fillText(v.toFixed(1), 2, Yz(v) + 4);
    }
    ctx.fillStyle = alpha(C.edge, 0.08);
    ctx.fillRect(Xz(900), split + 6, Xz(1000) - Xz(900), bottom - split - 6);
    ctx.fillStyle = alpha(C.mid, 0.6);
    for (let i = Z0; i < Z1; i++) { if (LOSS[i] < lo || LOSS[i] > hi) continue; ctx.fillRect(Xz(i + 1) - 1, Yz(LOSS[i]) - 1, 2, 2); }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    line(BASE, Xz, Yz, Z0, Z1);
    ctx.setLineDash([]);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    line(sm, Xz, Yz, Z0, Z1);
    ctx.fillStyle = C.mid;
    for (const s of [700, 800, 900, 1000]) { const t = String(s), tw = ctx.measureText(t).width; ctx.fillText(t, Math.min(W - PR - tw, Math.max(PL, Xz(s) - tw / 2)), bottom + 14); }
    ctx.fillText("확대", PL + 4, split + 18);
  }

  let raf = 0;
  function update() {
    sm = ema(LOSS, w, debias);
    const s = summarize(sm);
    ro.set("spike", `spike 최대 ${s.spikeRaw.toFixed(2)} → ${s.spikeSm.toFixed(2)}`, "hot");
    ro.set("lag", `900–1000 기저보다 ${s.lag.toFixed(3)} 낮게 (원본 ${s.lagRaw.toFixed(3)})`);
    ro.set("first", `첫 값 ${s.first.toFixed(3)} (원본 ${s.firstRaw.toFixed(3)})`, Math.abs(s.first - s.firstRaw) > 0.5 ? "hot" : "");
    ro.set("win", w > 0 ? `평균 창 ≈ ${Math.round(1 / (1 - w))} step` : "평균 창 1 step");
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(PRESETS) as PK[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => { w = PRESETS[k].w; pw.set(w); update(); }, "w6");
  const pw = param({ label: "w", min: 0, max: 0.995, step: 0.005, value: w, fmt: (v) => v.toFixed(3), aria: "smoothing 가중치 w", onInput: (v) => { w = v; pre.select(null); update(); } });
  const grid = el("div", "param-grid");
  const tg = el("label", "toggle");
  tg.innerHTML = `<input type="checkbox" checked aria-label="편향 보정" /> 보정 1/(1 − wᵏ)`;
  tg.querySelector("input")!.addEventListener("change", (e) => { debias = (e.target as HTMLInputElement).checked; update(); });
  grid.append(pw.row, tg);
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";   // 수업 시트 폭에서도 두 칸
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
