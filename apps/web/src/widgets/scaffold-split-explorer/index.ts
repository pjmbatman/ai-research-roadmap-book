// 무작위 분할 vs scaffold 분할 (data/widgets.toml: scaffold-split-explorer).
// 합성 "분자" 285개가 2차원 특성 공간에서 scaffold(골격) 24개로 무리 짓는다. 골격 크기는 실제처럼 치우쳐 있다(39개 ~ 7개).
// 물성 y = sin(1.1 x₁) + 0.35 x₂ + σ_s u_s + 0.1 ε에서 u_s는 골격마다 공유되는 효과(표준화)다.
// 모델은 RBF 커널 Gaussian process의 사후 평균(ℓ = 0.6, 잡음 분산 0.05, train 평균을 사전 평균으로)이다.
// 사후 평균은 y에 선형이라 분할마다 A = K*(K + λI)⁻¹을 한 번 구해 두고 σ_s를 바꿀 때는 곱만 다시 한다.
// 무작위 분할의 test 분자는 같은 골격의 친척이 train에 있어 최근접 유사도가 높고 오차가 작다.
// scaffold 분할은 처음 보는 골격에 대한 오차를 잰다. σ_s가 클수록 둘의 차이가 벌어진다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

const S = 24, ELL = 0.6, LAM = 0.05, SG_MAX = 1.5, CURVE = 31;
export const SIZES = Array.from({ length: S }, (_, k) => Math.round(36 * (k + 1) ** -0.7) + 3);
export const N = SIZES.reduce((a, b) => a + b, 0);
type Mode = "random" | "scaffold";

export interface Data { x1: Float64Array; x2: Float64Array; base: Float64Array; u: Float64Array; g: Int32Array; cx: number[]; cy: number[] }
export function makeData(): Data {
  const r = rng(28021);
  const cx: number[] = [], cy: number[] = [];
  // 5 × 5 격자에서 칸을 무작위로 섞어 골격 중심을 둔다(큰 골격이 한쪽에 몰리지 않게)
  const cells = Array.from({ length: 25 }, (_, i) => i);
  for (let i = 24; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [cells[i], cells[j]] = [cells[j], cells[i]]; }
  for (let k = 0; k < S; k++) {
    const c = cells[k];
    cx.push(-2.4 + 1.2 * (c % 5) + (r.u() - 0.5) * 0.5);
    cy.push(-2.4 + 1.2 * Math.floor(c / 5) + (r.u() - 0.5) * 0.5);
  }
  const u0 = Array.from({ length: S }, () => r.normal());
  const mu = u0.reduce((a, b) => a + b, 0) / S;
  const sd = Math.sqrt(u0.reduce((a, b) => a + (b - mu) ** 2, 0) / S);
  const d: Data = { x1: new Float64Array(N), x2: new Float64Array(N), base: new Float64Array(N), u: Float64Array.from(u0, (v) => (v - mu) / sd), g: new Int32Array(N), cx, cy };
  let n = 0;
  for (let k = 0; k < S; k++) for (let m = 0; m < SIZES[k]; m++, n++) {
    d.g[n] = k;
    d.x1[n] = cx[k] + 0.28 * r.normal();
    d.x2[n] = cy[k] + 0.28 * r.normal();
    d.base[n] = Math.sin(1.1 * d.x1[n]) + 0.35 * d.x2[n] + 0.1 * r.normal();
  }
  return d;
}

/** 시드로 test 집합을 고른다. 무작위는 분자 20%, scaffold는 골격을 섞어 test가 20%를 넘을 때까지 통째로 넣는다. */
export function split(mode: Mode, seed: number, d: Data): number[] {
  const r = rng(seed * 7919 + (mode === "scaffold" ? 11 : 5));
  const perm = (m: number) => { const a = Array.from({ length: m }, (_, i) => i); for (let i = m - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  if (mode === "random") return perm(N).slice(0, Math.floor(0.2 * N)).sort((a, b) => a - b);
  const held = new Set<number>();
  let cnt = 0;
  for (const s of perm(S)) { if (cnt >= 0.2 * N) break; held.add(s); cnt += SIZES[s]; }
  const te: number[] = [];
  for (let n = 0; n < N; n++) if (held.has(d.g[n])) te.push(n);
  return te;
}

const kern = (d: Data, a: number, b: number) => Math.exp(-((d.x1[a] - d.x1[b]) ** 2 + (d.x2[a] - d.x2[b]) ** 2) / (2 * ELL * ELL));

/** GP 사후 평균의 선형 사상 A = K*(K + λI)⁻¹ (n_te × n_tr)과 test 분자의 최근접 train 커널 유사도 */
export function fitSplit(d: Data, te: number[]) {
  const isTe = new Uint8Array(N); te.forEach((t) => (isTe[t] = 1));
  const tr: number[] = []; for (let n = 0; n < N; n++) if (!isTe[n]) tr.push(n);
  const m = tr.length;
  // Cholesky: K + λI = L Lᵀ
  const L = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
    let s = kern(d, tr[i], tr[j]) + (i === j ? LAM : 0);
    for (let k = 0; k < j; k++) s -= L[i * m + k] * L[j * m + k];
    L[i * m + j] = i === j ? Math.sqrt(s) : s / L[j * m + j];
  }
  // A의 각 행 a = (K + λI)⁻¹ k*  (대칭이므로 a = L⁻ᵀ L⁻¹ k*)
  const A = new Float64Array(te.length * m), sim = new Float64Array(te.length), z = new Float64Array(m);
  te.forEach((t, r) => {
    let mx = 0;
    for (let i = 0; i < m; i++) { const v = kern(d, t, tr[i]); z[i] = v; if (v > mx) mx = v; }
    sim[r] = mx;
    for (let i = 0; i < m; i++) { let s = z[i]; for (let k = 0; k < i; k++) s -= L[i * m + k] * z[k]; z[i] = s / L[i * m + i]; }
    for (let i = m - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < m; k++) s -= L[k * m + i] * z[k]; z[i] = s / L[i * m + i]; }
    A.set(z, r * m);
  });
  return { tr, te, A, sim };
}
export type Fit = ReturnType<typeof fitSplit>;

/** σ_s에서의 test MAE */
export function mae(d: Data, F: Fit, sg: number) {
  const y = (n: number) => d.base[n] + sg * d.u[d.g[n]];
  const m = F.tr.length;
  let mu = 0; for (const n of F.tr) mu += y(n); mu /= m;
  const ytr = F.tr.map((n) => y(n) - mu);
  let s = 0;
  F.te.forEach((t, r) => { let p = mu; for (let i = 0; i < m; i++) p += F.A[r * m + i] * ytr[i]; s += Math.abs(p - y(t)); });
  return s / F.te.length;
}
const median = (a: Float64Array) => { const b = [...a].sort((x, y) => x - y); const h = b.length >> 1; return b.length % 2 ? b[h] : (b[h - 1] + b[h]) / 2; };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "무작위 분할 vs scaffold 분할", "compare", narrow ? 1.4 : 0.56, "골격별로 무리 지은 분자의 특성 공간과 train·test 분할, 그리고 골격 효과 크기에 따른 두 분할의 test 오차 곡선");
  const data = makeData();
  let mode: Mode = "random", sg = 0.6, seed = 1;
  let fits: Record<Mode, Fit>, curves: Record<Mode, number[]>, sims: Record<Mode, number>;

  function refit() {
    fits = { random: fitSplit(data, split("random", seed, data)), scaffold: fitSplit(data, split("scaffold", seed, data)) };
    curves = { random: [], scaffold: [] };
    for (const k of ["random", "scaffold"] as Mode[]) for (let i = 0; i < CURVE; i++) curves[k].push(mae(data, fits[k], (i / (CURVE - 1)) * SG_MAX));
    sims = { random: median(fits.random.sim), scaffold: median(fits.scaffold.sim) };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const stacked = W < 460;
    const gap = 22, top = 24, pad = 10;
    const pw = stacked ? W - 2 * pad : (W - 2 * pad - gap) / 2;
    const ph = stacked ? (H - 2 * top - gap) / 2 : H - top - pad - 16;
    const side = Math.min(pw, ph);
    const A = stacked ? { x: (W - side) / 2, y: top } : { x: pad + (pw - side) / 2, y: top };
    const B = stacked ? { x: pad + 30, y: top * 2 + side + gap - 8, w: W - 2 * pad - 34, h: side - 18 } : { x: pad + pw + gap + 30, y: top, w: pw - 34, h: side - 18 };
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.textBaseline = "alphabetic";

    // 왼쪽: 특성 공간. scaffold 분할에서 test로 빠진 골격은 점선 원
    const te = new Uint8Array(N); fits[mode].te.forEach((t) => (te[t] = 1));
    const R = 3.3;
    const P = (x: number, y: number): [number, number] => [A.x + ((x + R) / (2 * R)) * side, A.y + ((R - y) / (2 * R)) * side];
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(A.x, A.y, side, side);
    ctx.fillStyle = C.mid; ctx.fillText("특성 공간 (x₁, x₂)", A.x, A.y - 8);
    if (mode === "scaffold") {
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
      const seen = new Set<number>();
      fits.scaffold.te.forEach((t) => { const k = data.g[t]; if (seen.has(k)) return; seen.add(k); const [px, py] = P(data.cx[k], data.cy[k]); ctx.beginPath(); ctx.arc(px, py, (0.72 / (2 * R)) * side, 0, 7); ctx.stroke(); });
      ctx.setLineDash([]);
    }
    for (const pass of [0, 1]) for (let n = 0; n < N; n++) {
      if (te[n] !== pass) continue;
      const [px, py] = P(data.x1[n], data.x2[n]);
      ctx.fillStyle = pass ? C.edge : C.ink2;
      ctx.beginPath(); ctx.arc(px, py, pass ? 3.2 : 2.2, 0, 7); ctx.fill();
    }

    // 오른쪽: σ_s에 따른 test MAE
    let hi = 0; for (const k of ["random", "scaffold"] as Mode[]) hi = Math.max(hi, ...curves[k]);
    hi = Math.ceil(hi * 4 + 0.2) / 4;
    const X = (v: number) => B.x + (v / SG_MAX) * B.w, Y = (v: number) => B.y + B.h - (v / hi) * B.h;
    ctx.fillStyle = C.mid; ctx.fillText("σ_s에 따른 test MAE", B.x - 30, B.y - 8);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(B.x, B.y, B.w, B.h);
    ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "right";
    for (let v = 0; v <= hi + 1e-9; v += hi > 1.5 ? 0.5 : 0.25) { ctx.fillText(v.toFixed(2), B.x - 4, Y(v) + 3); ctx.beginPath(); ctx.moveTo(B.x, Y(v)); ctx.lineTo(B.x + B.w, Y(v)); ctx.stroke(); }
    ctx.textAlign = "center";
    [0, 0.5, 1, 1.5].forEach((v) => ctx.fillText(v.toFixed(1), X(v), B.y + B.h + 13));
    ctx.textAlign = "start";
    // 지금 σ_s
    ctx.strokeStyle = C.mid; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(sg), B.y); ctx.lineTo(X(sg), B.y + B.h); ctx.stroke(); ctx.setLineDash([]);
    const col = (k: Mode) => (k === "random" ? C.ink2 : C.edge);
    for (const k of ["random", "scaffold"] as Mode[]) {
      ctx.strokeStyle = col(k); ctx.lineWidth = k === mode ? 2.8 : 1.6;
      ctx.beginPath(); curves[k].forEach((v, i) => { const x = X((i / (CURVE - 1)) * SG_MAX); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); }); ctx.stroke();
      const v = mae(data, fits[k], sg);
      ctx.fillStyle = col(k); ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(sg), Y(v), k === mode ? 5.5 : 4, 0, 7); ctx.fill(); ctx.stroke();
    }
    // 범례(왼쪽 위: 두 곡선 모두 σ_s = 0에서 낮다)
    ctx.font = "600 11px JetBrains Mono, monospace";
    (["scaffold", "random"] as Mode[]).forEach((k, i) => {
      const y = B.y + 16 + i * 16;
      ctx.strokeStyle = col(k); ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(B.x + 8, y - 4); ctx.lineTo(B.x + 24, y - 4); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.fillText(k === "random" ? "무작위" : "scaffold", B.x + 30, y);
    });
  }

  let raf = 0;
  function update() {
    const a = mae(data, fits.random, sg), b = mae(data, fits.scaffold, sg);
    ro.set("random", `무작위 MAE ${a.toFixed(3)}`, mode === "random" ? "hot" : "");
    ro.set("scaffold", `scaffold MAE ${b.toFixed(3)}`, mode === "scaffold" ? "hot" : "");
    ro.set("gap", `오차 비 ${(b / a).toFixed(1)}배`);
    ro.set("sim", `최근접 train 유사도 중앙값 ${sims[mode].toFixed(2)}`);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const modeBtns = presets<Mode>([{ key: "random", label: "무작위 분할" }, { key: "scaffold", label: "scaffold 분할" }], (k) => { mode = k; update(); }, mode);
  modeBtns.row.classList.add("segmented");
  const ps = param({ label: "σ_s", min: 0, max: SG_MAX, step: 0.05, value: sg, aria: "골격 효과의 표준편차 σ_s", onInput: (v) => { sg = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 분할";
  reseed.addEventListener("click", () => { seed++; refit(); update(); });
  tools.append(reseed);
  const ro = readouts();
  f.panel.append(modeBtns.row, grid, tools, ro.row);
  f.onResize(draw);
  refit();
  update();
}
