// 상태공간 모델 커널 (data/widgets.toml: ssm-kernel).
// 18.3장 "상태공간 모델": 연속시간 A = [[a, ω], [−ω, a]](고윳값 a ± iω), B = [1, 1]ᵀ, C = [1, 0], Δ = 0.1을
// zero-order hold로 이산화한다. 본문 셀처럼 블록 행렬 [[A, B], [0, 0]]·Δ의 행렬 지수 한 번으로 Ā와 B̄를 함께 얻는다.
// kernel K_j = C Ā^j B̄(j = 0..199)와, 같은 입력 u(고정 시드)에 대해 재귀 x_k = Ā x_{k−1} + B̄ u_k와 직접 합성곱 Σ K_j u_{k−j}를 비교한다.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";

export const DT = 0.1, L = 200;
const U = (() => { const r = rng(318); return Float64Array.from({ length: L }, () => r.normal()); })();

type M3 = number[][];
const mul = (P: M3, Q: M3) => P.map((row, i) => Q[0].map((_, j) => row.reduce((s, v, k) => s + v * Q[k][j], 0)));
/** 3×3 행렬 지수: scaling and squaring + Taylor 18항 */
export function expm3(M: M3): M3 {
  const nrm = Math.max(...M.map((r) => r.reduce((s, v) => s + Math.abs(v), 0)));
  const sq = Math.max(0, Math.ceil(Math.log2(nrm || 1)) + 1);
  const S = M.map((r) => r.map((v) => v / 2 ** sq));
  let term: M3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let E: M3 = term.map((r) => r.slice());
  for (let n = 1; n <= 18; n++) {
    term = mul(term, S).map((r) => r.map((v) => v / n));
    E = E.map((r, i) => r.map((v, j) => v + term[i][j]));
  }
  for (let i = 0; i < sq; i++) E = mul(E, E);
  return E;
}

export function discretize(a: number, om: number) {
  const M = expm3([[a * DT, om * DT, DT], [-om * DT, a * DT, DT], [0, 0, 0]]);
  return { Ab: [[M[0][0], M[0][1]], [M[1][0], M[1][1]]], Bb: [M[0][2], M[1][2]] };
}

export function simulate(a: number, om: number) {
  const { Ab, Bb } = discretize(a, om);
  const K = new Float64Array(L);
  let v = Bb.slice();
  for (let j = 0; j < L; j++) { K[j] = v[0]; v = [Ab[0][0] * v[0] + Ab[0][1] * v[1], Ab[1][0] * v[0] + Ab[1][1] * v[1]]; }
  const yRec = new Float64Array(L), yConv = new Float64Array(L);
  let x = [0, 0];
  for (let k = 0; k < L; k++) {
    x = [Ab[0][0] * x[0] + Ab[0][1] * x[1] + Bb[0] * U[k], Ab[1][0] * x[0] + Ab[1][1] * x[1] + Bb[1] * U[k]];
    yRec[k] = x[0];
    let s = 0;
    for (let j = 0; j <= k; j++) s += K[j] * U[k - j];
    yConv[k] = s;
  }
  let diff = 0;
  for (let k = 0; k < L; k++) diff = Math.max(diff, Math.abs(yRec[k] - yConv[k]));
  // Ā = e^{aΔ}·회전(ωΔ)이므로 |λ(Ā)| = e^{aΔ}, arg λ = ±ωΔ
  return { K, yRec, yConv, diff, mod: Math.exp(a * DT), arg: om * DT, Ab, Bb };
}

const PRESETS = {
  osc: { label: "−0.3±2i", a: -0.3, om: 2 },
  slow: { label: "−0.05", a: -0.05, om: 0 },
  fast: { label: "−1", a: -1, om: 0 },
  grow: { label: "+0.05±i", a: 0.05, om: 1 },
} as const;
type PK = keyof typeof PRESETS;

export function mount(root: HTMLElement) {
  const f = frame(root, "상태공간 모델 커널", "manipulate", 0.56, "이산 고윳값의 복소평면 위치, kernel K_j, 입력과 재귀·합성곱 출력");
  let a = -0.3, om = 2;
  let res = simulate(a, om);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    // 왼쪽: 복소평면에서 1 근처(Re 0.55–1.1, Im ±0.6)를 확대해 단위원과 λ(Ā)를 그린다
    const RE0 = 0.55, RE1 = 1.1, IM = 0.6;
    const ph = H - 34, pw = Math.round(ph * (RE1 - RE0) / (2 * IM));
    const px0 = 8, py0 = 14;
    const Zx = (re: number) => px0 + ((re - RE0) / (RE1 - RE0)) * pw, Zy = (im: number) => py0 + ((IM - im) / (2 * IM)) * ph;
    ctx.save();
    ctx.beginPath(); ctx.rect(px0, py0, pw, ph); ctx.clip();
    ctx.fillStyle = alpha(C.edge, 0.06);
    ctx.beginPath(); ctx.arc(Zx(0), Zy(0), Zx(1) - Zx(0), 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px0, Zy(0)); ctx.lineTo(px0 + pw, Zy(0)); ctx.stroke();
    ctx.strokeStyle = C.mid; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.arc(Zx(0), Zy(0), Zx(1) - Zx(0), 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    for (const sgn of [1, -1]) {
      const th = sgn * res.arg;
      ctx.fillStyle = res.mod > 1 ? C.ink : C.edge;
      ctx.beginPath(); ctx.arc(Zx(res.mod * Math.cos(th)), Zy(res.mod * Math.sin(th)), 5, 0, 7); ctx.fill();
    }
    ctx.restore();
    ctx.strokeStyle = C.faint; ctx.strokeRect(px0, py0, pw, ph);
    ctx.fillStyle = C.mid;
    ctx.fillText("1", Zx(1) - 3, py0 + ph + 13);
    ctx.fillText("0.6", Zx(0.6) - 9, py0 + ph + 13);
    ctx.fillText("λ(Ā)", px0 + 4, py0 + 12);
    const side = pw + 6;
    // 오른쪽: kernel(위)과 출력(아래)
    const x0 = px0 + side + 18, x1 = W - 8;
    const X = (j: number) => x0 + (j / (L - 1)) * (x1 - x0);
    const panel = (yTop: number, yBot: number, arrs: { v: Float64Array; col: string; w: number; dots?: boolean }[], label: string) => {
      let m = 1e-9; for (const { v } of arrs) for (const t of v) m = Math.max(m, Math.abs(t));
      const mid = (yTop + yBot) / 2, sc = (yBot - yTop) / 2 / (m * 1.08);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke();
      for (const { v, col, w, dots } of arrs) {
        if (dots) { ctx.fillStyle = col; for (let j = 0; j < L; j += 4) { ctx.beginPath(); ctx.arc(X(j), mid - v[j] * sc, 2.2, 0, 7); ctx.fill(); } continue; }
        ctx.strokeStyle = col; ctx.lineWidth = w;
        ctx.beginPath(); v.forEach((t, j) => (j ? ctx.lineTo(X(j), mid - t * sc) : ctx.moveTo(X(j), mid - t * sc))); ctx.stroke();
      }
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(x1 - tw - 6, yTop, tw + 6, 14);
      ctx.fillStyle = C.mid; ctx.fillText(label, x1 - tw - 3, yTop + 10);
    };
    const top = 8, split = H * 0.46, bottom = H - 20;
    panel(top, split - 6, [{ v: res.K, col: C.edge, w: 2 }], "kernel K_j");
    panel(split + 6, bottom, [{ v: U, col: alpha(C.mid, 0.45), w: 1 }, { v: res.yRec, col: C.ink2, w: 1.8 }, { v: res.yConv, col: C.edge, w: 0, dots: true }], "u, y (재귀 선·합성곱 점)");
    ctx.fillStyle = C.mid;
    for (const j of [0, 50, 100, 150, 199]) { const t = String(j), tw = ctx.measureText(t).width; ctx.fillText(t, Math.min(x1 - tw, Math.max(x0, X(j) - tw / 2)), bottom + 14); }
  }

  let raf = 0;
  function update() {
    res = simulate(a, om);
    ro.set("mod", `|λ(Ā)| ${res.mod.toFixed(4)}`, res.mod > 1 ? "hot" : "");
    ro.set("tau", a < 0 ? `시정수 ${(-1 / (a * DT)).toFixed(1)} step` : a === 0 ? "감쇠 없음" : `불안정 ${(1 / (a * DT)).toFixed(0)} step마다 e배`);
    ro.set("per", om > 0 ? `주기 ${(2 * Math.PI / (om * DT)).toFixed(1)} step` : "진동 없음");
    ro.set("K", `|K| j=0·50·199 ${[0, 50, 199].map((j) => Math.abs(res.K[j]).toFixed(4)).join("·")}`);
    ro.set("eq", `재귀−합성곱 ${res.diff.toExponential(1)}`, res.diff < 1e-8 ? "ok" : "hot");
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(PRESETS) as PK[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => { a = PRESETS[k].a; om = PRESETS[k].om; pa.set(a); po.set(om); update(); }, "osc");
  const pa = param({ label: "a", min: -1.5, max: 0.1, step: 0.01, value: a, fmt: (v) => v.toFixed(2), aria: "연속시간 고윳값의 실수부 a (감쇠)", onInput: (v) => { a = v; pre.select(null); update(); } });
  const po = param({ label: "ω", min: 0, max: 6, step: 0.05, value: om, fmt: (v) => v.toFixed(2), aria: "연속시간 고윳값의 허수부 ω (진동)", onInput: (v) => { om = v; pre.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pa.row, po.row);
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(140px, 1fr))";   // 수업 시트 폭에서도 두 칸
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
