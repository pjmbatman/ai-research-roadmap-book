// 흐릿함과 에너지 스펙트럼 (data/widgets.toml: energy-spectrum-blur).
// 27.4장의 합성 예를 그대로 쓴다: 위도권 한 바퀴 L = 128칸, 동서 파수 k = 1..63, 진폭 k^−1.5(E(k) ∝ k^−3),
// 파수별 예측 가능성 ρ_k = e^{−k/6}. 정답 o = ρa + √(1−ρ²) b, 날카로운 예보 = ρa + √(1−ρ²) b′(b′는 b의 다른 실현).
// 흐릿함 정도 β ∈ [0, 1]로 예보를 흐린다. 두 가지 방법:
//   · 예측 불가 성분 지우기(MSE 학습이 가는 방향): f = ρa + (1−β)√(1−ρ²) b′. β = 1이면 장의 "흐릿한 예보"(조건부 평균).
//   · 공간 평활: 날카로운 예보에 폭 σ = 8β 칸의 가우스 커널. 파수 공간에서 ĝ_k = exp(−½(2πkσ/L)²)를 곱한다.
// 사례 N = 200개(고정 시드)에서 RMSE·ACC·표준편차와 파수별 에너지 비 E_f(k)/E_o(k)를 Parseval로 정확히 계산한다.
// 격자 값 x_j = Σ_k 2Re(c_k e^{2πikj/L})이므로 격자 평균 x² = Σ_k 2|c_k|²다(장의 field 함수와 같은 규약).
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

const L = 128, K = 63, N = 200, SIG_MAX = 8;
const amp = Float64Array.from({ length: K }, (_, j) => (j + 1) ** -1.5);
export const rho = Float64Array.from({ length: K }, (_, j) => Math.exp(-(j + 1) / 6));
export type Mode = "mse" | "gauss";

/** 사례 N개의 복소 계수(N×K): 정답 o, 예측 가능한 부분 p = ρa, 예보의 예측할 수 없는 부분 u = √(1−ρ²) b′ */
export function makeData(seed = 27) {
  const r = rng(seed);
  const mk = () => { const re = new Float64Array(N * K), im = new Float64Array(N * K); for (let i = 0; i < N * K; i++) { const j = i % K; re[i] = (amp[j] * r.normal()) / Math.SQRT2; im[i] = (amp[j] * r.normal()) / Math.SQRT2; } return { re, im }; };
  const a = mk(), b = mk(), b2 = mk();
  const z = () => ({ re: new Float64Array(N * K), im: new Float64Array(N * K) });
  const o = z(), p = z(), u = z();
  for (let i = 0; i < N * K; i++) {
    const j = i % K, q = Math.sqrt(1 - rho[j] ** 2);
    p.re[i] = rho[j] * a.re[i]; p.im[i] = rho[j] * a.im[i];
    o.re[i] = p.re[i] + q * b.re[i]; o.im[i] = p.im[i] + q * b.im[i];
    u.re[i] = q * b2.re[i]; u.im[i] = q * b2.im[i];
  }
  return { o, p, u };
}
type Data = ReturnType<typeof makeData>;
export const gain = (k: number, sigma: number) => Math.exp(-0.5 * ((2 * Math.PI * k * sigma) / L) ** 2);
/** 파수 j(0부터)에서 예측 가능 성분과 예측 불가 성분에 곱하는 계수 */
const weights = (mode: Mode, beta: number, j: number): [number, number] => {
  if (mode === "mse") return [1, 1 - beta];
  const g = gain(j + 1, SIG_MAX * beta);
  return [g, g];
};

export function metrics(d: Data, mode: Mode, beta: number) {
  const wp = new Float64Array(K), wu = new Float64Array(K);
  for (let j = 0; j < K; j++) [wp[j], wu[j]] = weights(mode, beta, j);
  let mse = 0, accSum = 0, pf = 0, po = 0;
  const Ef = new Float64Array(K), Eo = new Float64Array(K);
  for (let n = 0; n < N; n++) {
    let e = 0, fo = 0, ff = 0, oo = 0;
    for (let j = 0; j < K; j++) {
      const i = n * K + j;
      const fr = wp[j] * d.p.re[i] + wu[j] * d.u.re[i], fi = wp[j] * d.p.im[i] + wu[j] * d.u.im[i], or = d.o.re[i], oi = d.o.im[i];
      e += 2 * ((fr - or) ** 2 + (fi - oi) ** 2);
      fo += 2 * (fr * or + fi * oi); ff += 2 * (fr * fr + fi * fi); oo += 2 * (or * or + oi * oi);
      Ef[j] += fr * fr + fi * fi; Eo[j] += or * or + oi * oi;
    }
    mse += e / N; accSum += fo / Math.sqrt(ff * oo); pf += ff / N; po += oo / N;
  }
  return { rmse: Math.sqrt(mse), acc: accSum / N, std: Math.sqrt(pf), stdTruth: Math.sqrt(po), ratio: Float64Array.from(Ef, (v, j) => v / Eo[j]) };
}

/** 사례 n의 격자 장 (L,) */
function fieldOf(d: Data, n: number, which: "truth" | "fc", mode: Mode, beta: number) {
  const x = new Float64Array(L);
  for (let j = 0; j < K; j++) {
    const i = n * K + j, k = j + 1;
    let cr: number, ci: number;
    if (which === "truth") { cr = d.o.re[i]; ci = d.o.im[i]; }
    else { const [wp, wu] = weights(mode, beta, j); cr = wp * d.p.re[i] + wu * d.u.re[i]; ci = wp * d.p.im[i] + wu * d.u.im[i]; }
    for (let q = 0; q < L; q++) { const th = (2 * Math.PI * k * q) / L; x[q] += 2 * (cr * Math.cos(th) - ci * Math.sin(th)); }
  }
  return x;
}

export const BETAS = Array.from({ length: 41 }, (_, i) => i / 40);

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400, wide = !narrow; // 수업 시트(약 419px)에서도 넓은 배치
  const f = frame(root, "흐릿함과 에너지 스펙트럼", "explore", wide ? 0.64 : 1.1,
    "정답과 흐린 예보의 위도권 단면, 파수별 에너지 비, 흐릿함 정도에 따른 RMSE 곡선");
  const d = makeData();
  const curves: Record<Mode, number[]> = { mse: BETAS.map((b) => metrics(d, "mse", b).rmse), gauss: BETAS.map((b) => metrics(d, "gauss", b).rmse) };
  const base = metrics(d, "mse", 0);
  const CASE = 3;
  const truthField = fieldOf(d, CASE, "truth", "mse", 0);
  let mode: Mode = "mse", beta = 0;
  let m = base;
  let fc = fieldOf(d, CASE, "fc", mode, beta);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const mono = "11px JetBrains Mono, monospace", sans = "600 12px Pretendard Variable, sans-serif";
    type Box = { x: number; y: number; w: number; h: number };
    let F: Box, S: Box, R: Box;
    if (wide) {
      const fh = Math.round(H * 0.3);
      F = { x: 40, y: 24, w: W - 52, h: fh - 24 };
      const by = fh + 30, bh = H - by - 24, sw = Math.round((W - 40) * 0.58);
      S = { x: 40, y: by, w: sw - 40, h: bh };
      R = { x: sw + 44, y: by, w: W - sw - 44 - 12, h: bh };
    } else {
      const fh = Math.round(H * 0.24), sh = Math.round(H * 0.4);
      F = { x: 40, y: 22, w: W - 52, h: fh - 22 };
      S = { x: 40, y: fh + 26, w: W - 52, h: sh - 30 };
      R = { x: 40, y: fh + sh + 34, w: W - 52, h: H - fh - sh - 34 - 22 };
    }
    const frameBox = (b: Box, title: string) => {
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      ctx.font = sans; ctx.fillStyle = C.mid; ctx.textAlign = "left"; ctx.fillText(title, b.x, b.y - 8);
    };
    // ── 위도권 단면
    frameBox(F, "위도권 한 바퀴(사례 하나)");
    let ext = 0; for (let q = 0; q < L; q++) ext = Math.max(ext, Math.abs(truthField[q]));
    ext *= 1.1;
    const XF = (q: number) => F.x + (q / (L - 1)) * F.w, YF = (v: number) => F.y + F.h / 2 - (v / ext) * (F.h / 2);
    const line = (x: Float64Array, col: string, lw: number) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); for (let q = 0; q < L; q++) (q ? ctx.lineTo(XF(q), YF(x[q])) : ctx.moveTo(XF(q), YF(x[q]))); ctx.stroke(); };
    ctx.save(); ctx.beginPath(); ctx.rect(F.x, F.y, F.w, F.h); ctx.clip();
    line(truthField, C.ink, 1.6);
    line(fc, C.edge, 2.2);
    ctx.restore();
    ctx.font = mono; ctx.textAlign = "right";
    const lg: [string, string][] = [["정답", C.ink], ["예보", C.edge]];
    let lx = F.x + F.w;
    for (let i = lg.length - 1; i >= 0; i--) {
      ctx.fillStyle = C.mid; ctx.fillText(lg[i][0], lx, F.y - 8);
      lx -= ctx.measureText(lg[i][0]).width + 5;
      ctx.strokeStyle = lg[i][1]; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx - 14, F.y - 12); ctx.lineTo(lx, F.y - 12); ctx.stroke();
      lx -= 22;
    }
    // ── 에너지 비
    frameBox(S, S.w < 260 ? "E_f(k)/E_o(k)" : "에너지 비 E_f(k)/E_o(k)");
    const kx = (k: number) => S.x + (Math.log(k) / Math.log(K)) * S.w;
    const ry = (v: number) => S.y + S.h - (Math.min(v, 1.4) / 1.4) * S.h;
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 0.5, 1]) { ctx.strokeStyle = v === 1 ? C.mid : C.faint; ctx.beginPath(); ctx.moveTo(S.x, ry(v)); ctx.lineTo(S.x + S.w, ry(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), S.x - 5, ry(v) + 4); }
    ctx.textAlign = "center";
    for (const k of [1, 2, 4, 8, 16, 32]) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(kx(k), S.y); ctx.lineTo(kx(k), S.y + S.h); ctx.stroke(); ctx.fillText(String(k), Math.max(kx(k), S.x + 6), S.y + S.h + 14); }
    ctx.save(); ctx.beginPath(); ctx.rect(S.x, S.y - 2, S.w, S.h + 2); ctx.clip();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5;
    ctx.beginPath(); for (let j = 0; j < K; j++) (j ? ctx.lineTo(kx(j + 1), ry(rho[j] ** 2)) : ctx.moveTo(kx(j + 1), ry(rho[j] ** 2))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = alpha(C.edge, 0.16); ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(kx(1), ry(0));
    for (let j = 0; j < K; j++) ctx.lineTo(kx(j + 1), ry(m.ratio[j]));
    ctx.lineTo(kx(K), ry(0)); ctx.closePath(); ctx.fill();
    ctx.beginPath(); for (let j = 0; j < K; j++) (j ? ctx.lineTo(kx(j + 1), ry(m.ratio[j])) : ctx.moveTo(kx(j + 1), ry(m.ratio[j]))); ctx.stroke();
    ctx.restore();
    ctx.font = mono; ctx.textAlign = "right"; ctx.fillStyle = C.mid;
    ctx.fillText("파수 k", S.x + S.w - 4, S.y + S.h - 6);
    const lgT = S.w < 260 ? "ρ_k²" : "ρ_k² 조건부 평균", lgW = ctx.measureText(lgT).width;
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(S.x + S.w - lgW - 26, S.y + 12); ctx.lineTo(S.x + S.w - lgW - 10, S.y + 12); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText(lgT, S.x + S.w - 4, S.y + 16);
    // ── RMSE(β): 두 방법
    frameBox(R, R.w < 260 ? "β에 따른 RMSE" : "흐릿함 정도에 따른 RMSE");
    const all = [...curves.mse, ...curves.gauss];
    const lo = Math.min(...all) * 0.96, hi = Math.max(...all) * 1.03;
    const sx = (b: number) => R.x + b * R.w, sy = (v: number) => R.y + R.h - ((v - lo) / (hi - lo)) * R.h;
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [Math.ceil(lo * 10) / 10, Math.floor(hi * 10) / 10]) ctx.fillText(v.toFixed(1), R.x - 5, sy(v) + 4);
    ctx.textAlign = "center";
    for (const b of [0, 0.5, 1]) ctx.fillText(b.toFixed(1), Math.min(Math.max(sx(b), R.x + 8), R.x + R.w - 10), R.y + R.h + 14);
    for (const k of ["gauss", "mse"] as Mode[]) {
      const on = k === mode;
      ctx.strokeStyle = on ? C.ink2 : alpha(C.mid, 0.6); ctx.lineWidth = on ? 2.2 : 1.4; ctx.setLineDash(on ? [] : [4, 3]);
      ctx.beginPath(); BETAS.forEach((b, i) => (i ? ctx.lineTo(sx(b), sy(curves[k][i])) : ctx.moveTo(sx(b), sy(curves[k][i])))); ctx.stroke();
      ctx.setLineDash([]);
      const yEnd = sy(curves[k][BETAS.length - 1]);
      ctx.fillStyle = on ? C.ink2 : C.mid; ctx.textAlign = "right";
      ctx.fillText(k === "mse" ? (R.w < 260 ? "지우기" : "예측 불가 성분 지우기") : (R.w < 260 ? "평활" : "공간 평활"), R.x + R.w - 6, k === "mse" ? yEnd - 6 : yEnd + 14);
    }
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(sx(beta), sy(m.rmse), 5.5, 0, 7); ctx.fill();
    ctx.textAlign = "left";
  }

  function update() {
    m = metrics(d, mode, beta);
    fc = fieldOf(d, CASE, "fc", mode, beta);
    const rel = (m.rmse / base.rmse - 1) * 100;
    ro.set("rmse", `RMSE ${m.rmse.toFixed(3)} (${rel >= 0 ? "+" : ""}${rel.toFixed(0)}%)`, rel < -0.5 ? "ok" : "");
    ro.set("acc", `ACC ${m.acc.toFixed(3)}`);
    ro.set("std", `표준편차 ${m.std.toFixed(2)} / 정답 ${m.stdTruth.toFixed(2)}`);
    ro.set("e", `에너지 비 k=8 ${m.ratio[7].toFixed(2)} · k=16 ${m.ratio[15].toFixed(2)}`, m.ratio[15] < 0.5 ? "hot" : "");
    pageState.set("energySpectrum", { mode, beta, sigma_cells: mode === "gauss" ? SIG_MAX * beta : null, rmse: m.rmse, acc: m.acc, std: m.std, ratio_k8: m.ratio[7], ratio_k16: m.ratio[15] }, root);
    draw();
  }

  const fmt = (v: number) => (mode === "gauss" ? `σ ${(SIG_MAX * v).toFixed(1)}칸` : v.toFixed(2));
  const pb = param({ label: "흐릿함 β", min: 0, max: 1, step: 0.025, value: beta, fmt, aria: "흐릿함 정도 (0 날카로운 예보, 1 가장 흐림)", onInput: (v) => { beta = v; update(); } });
  pb.row.style.gridTemplateColumns = "4.6em minmax(0, 1fr) 5.4em";
  const pre = presets<Mode>([{ key: "mse", label: "예측 불가 성분 지우기" }, { key: "gauss", label: "공간 평활(가우스)" }], (k) => { mode = k; pb.set(beta); update(); }, mode);
  pre.row.classList.add("segmented");
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 10px"));
  const grid = el("div", "param-grid");
  grid.append(pb.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
