// Metropolis 체인의 혼합 (data/widgets.toml: metropolis-chain).
// 11.7장의 예: 이중 우물 퍼텐셜 U(x) = (x² − 1)²의 Boltzmann 분포 p(x) ∝ exp(−U(x)/k_BT), k_BT = 0.15를
// 대칭 제안 x' = x + step·ε, ε ~ N(0, 1)과 수용 확률 min(1, p̃(x')/p̃(x))로 표집한다(x₀ = 1에서 시작, 2만 step).
// 잡음 ε와 균등 난수 u는 고정 시드이므로 step만 바꿔 같은 난수로 체인을 다시 돌린다.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const KT = 0.15, T = 20000, X_START = 1;
export const logTarget = (x: number) => -((x * x - 1) ** 2) / KT;
const NOISE = (() => {
  const r = rng(1953);
  const eps = new Float64Array(T), logu = new Float64Array(T);
  for (let t = 0; t < T; t++) { eps[t] = r.normal(); logu[t] = Math.log(Math.max(r.u(), 1e-300)); }
  return { eps, logu };
})();
/** 수치적분(격자) E[x²]와 정규화된 밀도 */
export const EXACT = (() => {
  const n = 40001, a = -4, b = 4, dx = (b - a) / (n - 1);
  let z = 0, m2 = 0;
  for (let i = 0; i < n; i++) { const x = a + i * dx, d = Math.exp(logTarget(x)); z += d; m2 += x * x * d; }
  return { ex2: m2 / z, Z: z * dx };
})();

export function runChain(step: number) {
  const chain = new Float64Array(T);
  let x = X_START, lp = logTarget(x), acc = 0;
  for (let t = 0; t < T; t++) {
    const prop = x + step * NOISE.eps[t], lq = logTarget(prop);
    if (NOISE.logu[t] < lq - lp) { x = prop; lp = lq; acc++; }
    chain[t] = x;
  }
  // 우물 사이 이동: x > 0.5면 오른쪽, x < −0.5면 왼쪽 우물에 들어간 것으로 보고, 우물이 바뀐 횟수를 센다.
  let well = X_START > 0 ? 1 : -1, hops = 0, right = 0, m2 = 0;
  for (let t = 0; t < T; t++) {
    const v = chain[t];
    if (v > 0.5 && well < 0) { well = 1; hops++; } else if (v < -0.5 && well > 0) { well = -1; hops++; }
    if (v > 0) right++;
    m2 += v * v;
  }
  return { chain, acc: acc / T, right: right / T, hops, ex2: m2 / T };
}

const PRESETS = { s005: { label: "step 0.05", step: 0.05 }, s05: { label: "step 0.5", step: 0.5 }, s2: { label: "step 2.0", step: 2 } } as const;
type PK = keyof typeof PRESETS;
const Y0 = -2, Y1 = 2;

export function mount(root: HTMLElement) {
  const f = frame(root, "Metropolis 혼합", "manipulate", 0.56, "이중 우물 목표분포를 Metropolis 체인으로 표집한 trace와 히스토그램");
  let step = 0.5;
  let res = runChain(step);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 28, top = 22, bottom = H - 24;
    const split = Math.round(W * (W < 520 ? 0.66 : 0.72));
    const Y = (v: number) => bottom - ((v - Y0) / (Y1 - Y0)) * (bottom - top);
    // 격자: 우물 x = ±1, 장벽 x = 0
    ctx.lineWidth = 1;
    for (const v of [-2, -1, 0, 1, 2]) {
      ctx.strokeStyle = v === 0 ? C.mid : C.faint;
      ctx.setLineDash(v === 0 ? [3, 4] : []);
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - 8, Y(v)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(String(v), v < 0 ? 4 : 10, Y(v) + 4);
    }
    // trace: 픽셀 열마다 최솟값~최댓값
    const tw = split - PL - 10;
    const cols = Math.max(1, Math.floor(tw));
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 0; c < cols; c++) {
      const a = Math.floor((c * T) / cols), b = Math.max(a + 1, Math.floor(((c + 1) * T) / cols));
      let lo = Infinity, hi = -Infinity;
      for (let t = a; t < b; t++) { const v = res.chain[t]; if (v < lo) lo = v; if (v > hi) hi = v; }
      const x = PL + c + 0.5;
      ctx.moveTo(x, Y(Math.min(Y1, hi)) - 0.5); ctx.lineTo(x, Y(Math.max(Y0, lo)) + 0.5);
    }
    ctx.stroke();
    ctx.fillStyle = C.mid;
    ctx.fillText("x", PL + 4, top - 8);
    for (const t of [0, 10000, 20000]) {
      const s = t === 0 ? "0" : `${t / 1000}k`;
      const x = PL + (t / T) * tw;
      ctx.fillText(s, Math.min(split - 10 - ctx.measureText(s).width, Math.max(PL, x - ctx.measureText(s).width / 2)), bottom + 15);
    }
    // 히스토그램과 참 밀도(가로 막대)
    const hx0 = split + 6, hx1 = W - 8;
    const NB = 60, cnt = new Array(NB).fill(0);
    for (let t = 0; t < T; t++) { const v = res.chain[t]; if (v >= Y0 && v < Y1) cnt[Math.floor(((v - Y0) / (Y1 - Y0)) * NB)]++; }
    const db = (Y1 - Y0) / NB;
    const dens = cnt.map((c) => c / (T * db));
    const pmax = Math.exp(0) / EXACT.Z;
    const xmax = Math.max(pmax, ...dens) * 1.05;
    const Xh = (d: number) => hx0 + (d / xmax) * (hx1 - hx0);
    ctx.fillStyle = alpha(C.edge, 0.4);
    dens.forEach((d, b) => { if (!d) return; const y1 = Y(Y0 + (b + 1) * db), y0 = Y(Y0 + b * db); ctx.fillRect(hx0, y1 + 0.3, Xh(d) - hx0, y0 - y1 - 0.6); });
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const v = Y0 + ((Y1 - Y0) * i) / 200, d = Math.exp(logTarget(v)) / EXACT.Z; i ? ctx.lineTo(Xh(d), Y(v)) : ctx.moveTo(Xh(d), Y(v)); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx0, top); ctx.lineTo(hx0, bottom); ctx.stroke();
    ctx.fillStyle = C.mid;
    const lab = "p(x)";
    ctx.fillText(lab, hx1 - ctx.measureText(lab).width, top - 8);
    const lab2 = "step t";
    ctx.fillText(lab2, split - 10 - ctx.measureText(lab2).width, top - 8);
    // 출발점
    ctx.fillStyle = alpha(C.good, 0.9);
    ctx.beginPath(); ctx.arc(PL + 1, Y(X_START), 3.5, 0, 7); ctx.fill();
  }

  let raf = 0;
  function update() {
    res = runChain(step);
    ro.set("acc", `수용률 ${res.acc.toFixed(2)}`);
    ro.set("right", `오른쪽 우물 비율 ${res.right.toFixed(3)} (목표 0.5)`);
    ro.set("hops", `우물 사이 이동 ${res.hops}번`, "hot");
    ro.set("ex2", `E[x²] ${res.ex2.toFixed(4)} (수치적분 ${EXACT.ex2.toFixed(4)})`);
    pageState.set("metropolis", { step, acceptance: res.acc }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(PRESETS) as PK[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => { step = PRESETS[k].step; ps.set(Math.log10(step)); update(); }, "s05");
  const ps = param({ label: "step", min: -2, max: 0.7, step: 0.01, value: Math.log10(step), fmt: (v) => { const s = Math.pow(10, v); return s < 0.1 ? s.toFixed(3) : s.toFixed(2); }, aria: "제안 step 크기 (로그 눈금)", onInput: (v) => { step = Math.pow(10, v); pre.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
