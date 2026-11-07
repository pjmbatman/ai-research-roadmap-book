// 대조학습 붕괴 (data/widgets.toml: contrastive-collapse).
// 19.8장의 예: 표본 N개, 차원 D = 16. 건강한 표현은 표본마다 고유한 방향 base에 view마다 0.2 크기 잡음을 더한 것이다.
// 붕괴 정도 c로 건강한 표현과 붕괴한 표현을 섞는다: z = (1 − c)·건강 + c·붕괴.
//   완전 붕괴: 1 벡터 + 1e−3 잡음(모든 입력이 거의 같은 벡터), 차원 붕괴: 3차원 부분공간의 표현 + 0.05 잡음.
// 지표는 장의 코드와 같다: NT-Xent(2N개 view, 자기 자신을 뺀 2N − 1개 중 짝 고르기), 정렬 MSE mean((z − z′)²),
// 유효 rank(중심화한 z의 특이값 분포 엔트로피의 지수), 분산 벌점 mean(max(0, 1 − √(var + 1e−4))).
// 난수는 고정 시드라 c를 바꿔도 같은 표본을 쓴다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

export const D = 16, NMAX = 256, NS = [16, 32, 64, 128, 256], CS = Array.from({ length: 11 }, (_, i) => i / 10);
export type Mode = "full" | "dim";

export function makeData(seed = 2020) {
  const r = rng(seed);
  const mat = (n: number, m: number, s = 1) => Float64Array.from({ length: n * m }, () => s * r.normal());
  const base = mat(NMAX, D), e1 = mat(NMAX, D, 0.2), e2 = mat(NMAX, D, 0.2);
  const g = mat(NMAX, 3), P = mat(3, D), low = new Float64Array(NMAX * D);
  for (let n = 0; n < NMAX; n++) for (let j = 0; j < D; j++) low[n * D + j] = g[n * 3] * P[j] + g[n * 3 + 1] * P[D + j] + g[n * 3 + 2] * P[2 * D + j];
  const l1 = mat(NMAX, D, 0.05), l2 = mat(NMAX, D, 0.05), c1 = mat(NMAX, D, 1e-3), c2 = mat(NMAX, D, 1e-3);
  return { base, e1, e2, low, l1, l2, c1, c2 };
}
type Data = ReturnType<typeof makeData>;

/** 두 view (N × D) */
export function views(d: Data, mode: Mode, c: number, N: number) {
  const z1 = new Float64Array(N * D), z2 = new Float64Array(N * D);
  for (let k = 0; k < N * D; k++) {
    const t1 = mode === "full" ? 1 + d.c1[k] : d.low[k] + d.l1[k];
    const t2 = mode === "full" ? 1 + d.c2[k] : d.low[k] + d.l2[k];
    z1[k] = (1 - c) * (d.base[k] + d.e1[k]) + c * t1;
    z2[k] = (1 - c) * (d.base[k] + d.e2[k]) + c * t2;
  }
  return { z1, z2 };
}

export function ntXent(z1: Float64Array, z2: Float64Array, N: number, tau: number) {
  const M = 2 * N, z = new Float64Array(M * D);
  for (let i = 0; i < M; i++) {
    const src = i < N ? z1 : z2, o = (i % N) * D;
    let s = 0; for (let j = 0; j < D; j++) s += src[o + j] ** 2;
    const inv = 1 / Math.sqrt(s);
    for (let j = 0; j < D; j++) z[i * D + j] = src[o + j] * inv;
  }
  const S = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let k = i + 1; k < M; k++) {
    let s = 0; for (let j = 0; j < D; j++) s += z[i * D + j] * z[k * D + j];
    S[i * M + k] = S[k * M + i] = s / tau;
  }
  let loss = 0;
  for (let i = 0; i < M; i++) {
    let mx = -Infinity;
    for (let k = 0; k < M; k++) if (k !== i && S[i * M + k] > mx) mx = S[i * M + k];
    let se = 0;
    for (let k = 0; k < M; k++) if (k !== i) se += Math.exp(S[i * M + k] - mx);
    const pos = i < N ? i + N : i - N;
    loss += Math.log(se) + mx - S[i * M + pos];
  }
  return loss / M;
}

/** 16 × 16 대칭 행렬의 고윳값(순환 Jacobi) */
function eigvals(Ain: Float64Array, m: number) {
  const A = Float64Array.from(Ain);
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0, diag = 0;
    for (let p = 0; p < m; p++) { diag += A[p * m + p] ** 2; for (let q = p + 1; q < m; q++) off += A[p * m + q] ** 2; }
    if (off <= 1e-30 * diag) break;
    for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++) {
      const apq = A[p * m + q];
      if (apq === 0) continue;
      const th = (A[q * m + q] - A[p * m + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < m; k++) { const a = A[k * m + p], b = A[k * m + q]; A[k * m + p] = c * a - s * b; A[k * m + q] = s * a + c * b; }
      for (let k = 0; k < m; k++) { const a = A[p * m + k], b = A[q * m + k]; A[p * m + k] = c * a - s * b; A[q * m + k] = s * a + c * b; }
    }
  }
  return Array.from({ length: m }, (_, i) => A[i * m + i]);
}

export function stats(z1: Float64Array, z2: Float64Array, N: number) {
  const mean = new Float64Array(D), v = new Float64Array(D);
  for (let n = 0; n < N; n++) for (let j = 0; j < D; j++) mean[j] += z1[n * D + j] / N;
  const G = new Float64Array(D * D);
  let align = 0;
  for (let n = 0; n < N; n++) for (let j = 0; j < D; j++) {
    const a = z1[n * D + j] - mean[j];
    v[j] += (a * a) / N;
    align += (z1[n * D + j] - z2[n * D + j]) ** 2;
    for (let k = j; k < D; k++) G[j * D + k] += a * (z1[n * D + k] - mean[k]);
  }
  for (let j = 0; j < D; j++) for (let k = 0; k < j; k++) G[j * D + k] = G[k * D + j];
  const s = eigvals(G, D).map((e) => Math.sqrt(Math.max(e, 0)));
  const tot = s.reduce((a, b) => a + b, 0);
  let H = 0; for (const x of s) { const p = x / tot; if (p > 0) H -= p * Math.log(p); }
  let pen = 0; for (let j = 0; j < D; j++) pen += Math.max(0, 1 - Math.sqrt(v[j] + 1e-4)) / D;
  return { align: align / (N * D), effRank: Math.exp(H), varPen: pen };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "대조학습 붕괴", "manipulate", narrow ? 0.85 : 0.55, "붕괴 정도에 따른 NT-Xent, 정렬 MSE, 유효 rank, 분산 벌점 곡선 네 개");
  const data = makeData();
  let mode: Mode = "full", c = 0, tau = 0.1, ni = 4;
  const N = () => NS[ni];
  type Curves = { loss: number[]; align: number[]; rank: number[]; pen: number[] };
  const curves: Record<Mode, Curves> = { full: { loss: [], align: [], rank: [], pen: [] }, dim: { loss: [], align: [], rank: [], pen: [] } };
  let cur = { loss: 0, align: 0, effRank: 0, varPen: 0 };

  function computeCurves(m: Mode, withStats: boolean) {
    const cv = curves[m];
    cv.loss = [];
    if (withStats) { cv.align = []; cv.rank = []; cv.pen = []; }
    for (const cc of CS) {
      const { z1, z2 } = views(data, m, cc, N());
      cv.loss.push(ntXent(z1, z2, N(), tau));
      if (withStats) { const s = stats(z1, z2, N()); cv.align.push(s.align); cv.rank.push(s.effRank); cv.pen.push(s.varPen); }
    }
  }
  function computeCurrent() {
    const { z1, z2 } = views(data, mode, c, N());
    cur = { loss: ntXent(z1, z2, N(), tau), ...stats(z1, z2, N()) };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const lnN = Math.log(2 * N() - 1);
    const panels: { title: string; key: keyof Curves; lo: number; hi: number; now: number; ref?: number; refLabel?: string }[] = [
      { title: `NT-Xent (τ = ${tau.toFixed(2)})`, key: "loss", lo: 0, hi: Math.max(lnN, ...curves.full.loss, ...curves.dim.loss) * 1.08, now: cur.loss, ref: lnN, refLabel: "log(2N−1)" },
      { title: "정렬 MSE", key: "align", lo: 0, hi: Math.max(...curves.full.align, ...curves.dim.align) * 1.15, now: cur.align },
      { title: "유효 rank", key: "rank", lo: 0, hi: D * 1.1, now: cur.effRank, ref: D, refLabel: "D" },
      { title: "분산 벌점", key: "pen", lo: 0, hi: 1.1, now: cur.varPen },
    ];
    const gx = 8, gy = 6, cw = (W - 3 * gx) / 2, ch = (H - 3 * gy) / 2;
    panels.forEach((p, i) => {
      const ox = gx + (i % 2) * (cw + gx), oy = gy + Math.floor(i / 2) * (ch + gy);
      const PL = ox + 34, PR = ox + cw - 6, top = oy + 22, bottom = oy + ch - 18;
      const X = (v: number) => PL + v * (PR - PL);
      const Y = (v: number) => bottom - ((v - p.lo) / (p.hi - p.lo)) * (bottom - top);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.strokeRect(ox + 0.5, oy + 0.5, cw - 1, ch - 1);
      ctx.beginPath(); ctx.moveTo(PL, bottom); ctx.lineTo(PR, bottom); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
      ctx.fillText(p.title, ox + 8, oy + 15);
      ctx.fillStyle = C.mid; ctx.font = "10px JetBrains Mono, monospace";
      ctx.fillText("c = 0", X(0), bottom + 12); ctx.fillText("1", X(1) - 6, bottom + 12);
      const hiTxt = p.hi >= 10 ? p.hi.toFixed(0) : p.hi >= 1 ? p.hi.toFixed(1) : p.hi.toFixed(2);
      ctx.fillText(hiTxt, PL - 6 - ctx.measureText(hiTxt).width, top + 4);
      if (p.ref !== undefined) {
        ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
        ctx.beginPath(); ctx.moveTo(PL, Y(p.ref)); ctx.lineTo(PR, Y(p.ref)); ctx.stroke(); ctx.setLineDash([]);
        const t = p.key === "rank" ? `D = ${D}` : `${p.refLabel} ${p.ref.toFixed(2)}`;
        ctx.fillText(t, PR - ctx.measureText(t).width, Y(p.ref) - 4);
      }
      for (const m of [mode === "full" ? "dim" : "full", mode] as Mode[]) {
        const ys = curves[m][p.key];
        ctx.strokeStyle = m === mode ? C.edge : alpha(C.ink2, 0.5);
        ctx.lineWidth = m === mode ? 2.2 : 1.4;
        ctx.beginPath(); ys.forEach((v, k) => (k ? ctx.lineTo(X(CS[k]), Y(v)) : ctx.moveTo(X(CS[k]), Y(v)))); ctx.stroke();
      }
      ctx.fillStyle = C.edge;
      ctx.beginPath(); ctx.arc(X(c), Y(p.now), 4.5, 0, 7); ctx.fill();
    });
  }

  // 계산 수준 1: 현재 점, 2: + 손실 곡선, 3: + 모든 곡선. 고른 계열을 먼저 그리고, 다른 계열은 다음 프레임에 채운다.
  let raf = 0, pending = 0, other = 0;
  function schedule(level: number) {
    pending = Math.max(pending, level);
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const lv = pending;
      pending = 0;
      if (lv >= 2) { computeCurves(mode, lv >= 3); other = Math.max(other, lv); }
      computeCurrent();
      const lnN = Math.log(2 * N() - 1);
      ro.set("loss", `NT-Xent ${cur.loss.toFixed(3)}`, Math.abs(cur.loss - lnN) < 0.05 ? "hot" : "");
      ro.set("align", `정렬 ${cur.align.toFixed(4)}`);
      ro.set("rank", `rank ${cur.effRank.toFixed(2)}`, cur.effRank < 8 ? "hot" : "");
      ro.set("pen", `벌점 ${cur.varPen.toFixed(3)}`, cur.varPen > 0.5 ? "hot" : "");
      pageState.set("contrastive", { mode, c, tau, N: N(), ...cur }, root);
      if (curves.full.loss.length === CS.length && curves.dim.loss.length === CS.length) draw();
      if (other && !pending) {
        requestAnimationFrame(() => {
          if (!other || pending) return;
          computeCurves(mode === "full" ? "dim" : "full", other >= 3);
          other = 0;
          draw();
        });
      }
    });
  }

  const pre = presets<Mode>([{ key: "full", label: "완전 붕괴" }, { key: "dim", label: "차원 붕괴(3차원)" }], (k) => { mode = k; schedule(1); }, mode);
  pre.row.classList.add("segmented");
  const pc = param({ label: "붕괴", min: 0, max: 1, step: 0.05, value: c, fmt: (v) => v.toFixed(2), aria: "붕괴 정도 c (0 건강, 1 붕괴)", onInput: (v) => { c = v; schedule(1); } });
  const pt = param({ label: "τ", min: 0.05, max: 1, step: 0.05, value: tau, fmt: (v) => v.toFixed(2), aria: "temperature τ", onInput: (v) => { tau = v; schedule(2); } });
  const pn = param({ label: "N", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => String(NS[v]), aria: "batch 표본 수 N", onInput: (v) => { ni = v; pn.input.setAttribute("aria-valuetext", String(NS[v])); schedule(3); } });
  pn.input.setAttribute("aria-valuetext", String(NS[ni]));
  const grid = el("div", "param-grid");
  grid.append(pc.row, pt.row, pn.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(() => { if (curves.full.loss.length) draw(); });
  schedule(3);
}
