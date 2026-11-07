// 중요도 가중과 유효 표본 크기 (data/widgets.toml: importance-weights).
// 11.6장의 예: 목표 p = N(0, 1)에서 E_p[X²] = 1을 제안분포 q = N(μ_q, σ_q²)의 표본으로 추정한다.
// 자기 정규화 가중치 w̃_i = w_i / Σ w, w_i = p(x_i) / q(x_i)로 Î = Σ w̃_i x_i², ESS = (Σ w)² / Σ w².
// 같은 설정을 R번 반복해(고정 시드) 추정값이 어떻게 흩어지는지 본다. x_i = μ_q + σ_q z_i 이므로
//   log w_i = −x_i²/2 + z_i²/2 + log σ_q.
// 가우시안끼리의 E_q[w²] = σ_q² / √(2σ_q² − 1) · exp(μ_q² / (2σ_q² − 1)) (σ_q² > 1/2), 아니면 ∞. N → ∞에서 ESS/N → 1 / E_q[w²].
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const N = 1000, R = 200;
/** 고정 시드 표준정규 표본 z, (R·N,) */
export const Z = (() => { const r = rng(20261107); const z = new Float64Array(R * N); for (let i = 0; i < z.length; i++) z[i] = r.normal(); return z; })();

/** 반복마다 자기 정규화 추정값과 ESS/N, 그리고 첫 실행의 표본·정규화 가중치 */
export function simulate(mu: number, sigma: number) {
  const est = new Float64Array(R), ess = new Float64Array(R), wmax = new Float64Array(R);
  const x0 = new Float64Array(N), w0 = new Float64Array(N);
  const lw = new Float64Array(N);
  const ls = Math.log(sigma);
  for (let r = 0; r < R; r++) {
    let m = -Infinity;
    for (let i = 0; i < N; i++) {
      const z = Z[r * N + i], x = mu + sigma * z;
      lw[i] = -0.5 * x * x + 0.5 * z * z + ls;
      if (lw[i] > m) m = lw[i];
    }
    let s = 0, s2 = 0, sf = 0, mx = 0;
    for (let i = 0; i < N; i++) {
      const w = Math.exp(lw[i] - m), x = mu + sigma * Z[r * N + i];
      s += w; s2 += w * w; sf += w * x * x;
      if (w > mx) mx = w;
      if (r === 0) { x0[i] = x; w0[i] = w; }
    }
    est[r] = sf / s; ess[r] = (s * s) / s2 / N; wmax[r] = mx / s;
    if (r === 0) for (let i = 0; i < N; i++) w0[i] /= s;
  }
  return { est, ess, wmax, x0, w0 };
}
/** E_q[w²] (가중치 분산이 유한한지): σ_q² ≤ 1/2이면 ∞ */
export function secondMoment(mu: number, sigma: number) {
  const d = 2 * sigma * sigma - 1;
  return d <= 0 ? Infinity : ((sigma * sigma) / Math.sqrt(d)) * Math.exp((mu * mu) / d);
}
const quant = (a: Float64Array, q: number) => { const s = Array.from(a).sort((u, v) => u - v); const k = (s.length - 1) * q, i = Math.floor(k); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (k - i); };

const PRESETS = {
  wide: { label: "꼬리 두꺼움", mu: 0, sigma: 2 },
  shift: { label: "위치 어긋남", mu: 2, sigma: 1 },
  narrow: { label: "꼬리 얇음", mu: 0, sigma: 0.5 },
} as const;
type PK = keyof typeof PRESETS;

const X0 = -5, X1 = 6, E0 = 0, E1 = 2.5, BINS = 50;
const gauss = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

export function mount(root: HTMLElement) {
  const f = frame(root, "중요도 가중과 ESS", "manipulate", 0.7, "목표분포와 제안분포, 첫 실행 표본의 정규화 가중치, 그리고 200번 반복한 추정값의 히스토그램");
  let mu: number = PRESETS.wide.mu, sigma: number = PRESETS.wide.sigma;
  let sim = simulate(mu, sigma);

  const PAD_L = 12, PAD_R = 12;
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const top = 24, base1 = Math.round(H * 0.56), top2 = base1 + 46, base2 = H - 22;
    const X = (x: number) => PAD_L + ((x - X0) / (X1 - X0)) * (W - PAD_L - PAD_R);
    // 위: 밀도와 가중치
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, base1); ctx.lineTo(W - PAD_R, base1); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (let k = -4; k <= 6; k += 2) { const s = String(k); ctx.fillText(s, X(k) - ctx.measureText(s).width / 2, base1 + 14); }
    // 막대 높이: 가장 큰 가중치에 맞추되, 가중치가 고르면(모두 1/N 근처) 낮게 깔리도록 25/N을 하한으로 둔다
    const wm = Math.max(...sim.w0, 25 / N);
    const hW = base1 - top - 4;
    ctx.strokeStyle = alpha(C.edge, 0.7); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = sim.x0[i];
      if (x < X0 || x > X1) continue;
      const h = (sim.w0[i] / wm) * hW;
      if (h < 0.4) continue;
      ctx.moveTo(X(x), base1); ctx.lineTo(X(x), base1 - h);
    }
    ctx.stroke();
    const xs = Array.from({ length: 301 }, (_, i) => X0 + ((X1 - X0) * i) / 300);
    const dmax = Math.max(gauss(0, 0, 1), gauss(mu, mu, sigma));
    const Yd = (v: number) => base1 - (v / dmax) * hW * 0.92;
    const curve = (m: number, s: number) => { ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo(X(x), Yd(gauss(x, m, s))) : ctx.moveTo(X(x), Yd(gauss(x, m, s))))); };
    ctx.fillStyle = C.fill; curve(mu, sigma); ctx.lineTo(X(X1), base1); ctx.lineTo(X(X0), base1); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2; curve(mu, sigma); ctx.stroke();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); curve(0, 1); ctx.stroke(); ctx.setLineDash([]);
    // 범례
    const items: [string, string, "dash" | "line" | "stem"][] = [["목표 p", C.ink, "dash"], ["제안 q", C.ink2, "line"], ["가중치 w̃ (첫 실행)", C.edge, "stem"]];
    let lx = PAD_L + 4;
    for (const [name, col, kind] of items) {
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      if (kind === "stem") { ctx.beginPath(); ctx.moveTo(lx + 8, 18); ctx.lineTo(lx + 8, 6); ctx.stroke(); }
      else { ctx.setLineDash(kind === "dash" ? [4, 3] : []); ctx.beginPath(); ctx.moveTo(lx, 12); ctx.lineTo(lx + 16, 12); ctx.stroke(); ctx.setLineDash([]); }
      ctx.fillStyle = C.mid; ctx.fillText(name, lx + 20, 16);
      lx += 30 + ctx.measureText(name).width;
    }
    // 아래: 추정값 히스토그램
    const XE = (v: number) => PAD_L + ((v - E0) / (E1 - E0)) * (W - PAD_L - PAD_R);
    const cnt = new Array(BINS).fill(0);
    let over = 0;
    for (const v of sim.est) { if (v >= E1) { over++; cnt[BINS - 1]++; } else cnt[Math.max(0, Math.floor(((v - E0) / (E1 - E0)) * BINS))]++; }
    const cm = Math.max(...cnt, 1);
    const bw = (W - PAD_L - PAD_R) / BINS;
    ctx.fillStyle = alpha(C.mid, 0.55);
    cnt.forEach((c, b) => { if (!c) return; const h = (c / cm) * (base2 - top2 - 6); ctx.fillRect(PAD_L + b * bw + 0.5, base2 - h, bw - 1, h); });
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, base2); ctx.lineTo(W - PAD_R, base2); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (const k of [0, 0.5, 1, 1.5, 2, 2.5]) { const s = String(k); ctx.fillText(s, Math.min(W - PAD_R - ctx.measureText(s).width, Math.max(PAD_L, XE(k) - ctx.measureText(s).width / 2)), base2 + 14); }
    ctx.fillText(`추정값 Î (${R}번 반복)`, PAD_L + 2, top2 - 10);
    if (over) { const s = `≥ ${E1}: ${over}번`; ctx.fillText(s, W - PAD_R - ctx.measureText(s).width, top2 - 10); }
    ctx.strokeStyle = C.good; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(XE(1), top2 - 4); ctx.lineTo(XE(1), base2); ctx.stroke();
    const med = quant(sim.est, 0.5);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(XE(Math.min(med, E1)), top2 - 4); ctx.lineTo(XE(Math.min(med, E1)), base2); ctx.stroke();
  }

  let raf = 0;
  function update() {
    sim = simulate(mu, sigma);
    const m2 = secondMoment(mu, sigma);
    const med = quant(sim.est, 0.5), lo = quant(sim.est, 0.05), hi = quant(sim.est, 0.95);
    ro.set("ess", `ESS/N 첫 실행 ${sim.ess[0].toFixed(3)} · 중앙값 ${quant(sim.ess, 0.5).toFixed(3)}`, "hot");
    ro.set("wmax", `최대 가중치 비중 ${(sim.wmax[0] * 100).toFixed(1)}%`);
    ro.set("est", `Î 중앙값 ${med.toFixed(3)} · 90% ${lo.toFixed(2)}~${hi.toFixed(2)}`, Math.abs(med - 1) < 0.02 && hi - lo < 0.2 ? "ok" : "");
    ro.set("m2", Number.isFinite(m2) ? `E_q[w²] = ${m2 < 1e4 ? m2.toFixed(2) : m2.toExponential(1)} → ESS/N ${(1 / m2).toFixed(3)}` : "E_q[w²] = ∞ (σ_q² ≤ 1/2)");
    pageState.set("proposal", { mu, sigma }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(PRESETS) as PK[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => {
    mu = PRESETS[k].mu; sigma = PRESETS[k].sigma; pm.set(mu); ps.set(Math.log2(sigma)); update();
  }, "wide");
  const pm = param({ label: "μ_q", min: -3, max: 4, step: 0.1, value: mu, fmt: (v) => v.toFixed(1), aria: "제안분포 평균 μ_q", onInput: (v) => { mu = v; pre.select(null); update(); } });
  const ps = param({ label: "σ_q", min: -2, max: 1.6, step: 0.05, value: Math.log2(sigma), fmt: (v) => Math.pow(2, v).toFixed(2), aria: "제안분포 표준편차 σ_q (로그 눈금)", onInput: (v) => { sigma = Math.pow(2, v); pre.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pm.row, ps.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
