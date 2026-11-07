// 정규화 경로 (data/widgets.toml: regularization-path).
// 16.2장의 당뇨병 데이터(442명, 표준화한 특징 10개)에서 강도 α를 바꾸며 계수와 5-fold CV 오차를 본다.
//   Lasso: min (1/2N)‖y − Xw − b‖² + α‖w‖₁  (좌표 하강법, 절편은 벌점 없음)
//   Ridge: min ‖y − Xw − b‖² + λ‖w‖²,  λ = 2Nα  (본문 셀과 같은 척도: Ridge(alpha=2·N·α))
// 계산은 충분통계량(XᵀX, Xᵀy, Σx, Σy)만으로 한다. CV의 학습 fold 통계 = 전체 − 검증 fold.
// α 격자는 LassoCV와 같다: α_max = max|Xᵀ(y − ȳ)|/N에서 α_max·10⁻³까지 로그 간격 100개.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { NAMES, FOLDS, type Fold } from "./data";

const P = NAMES.length;
type Stats = { n: number; sx: number[]; sy: number; xx: number[][]; xy: number[]; yy: number };
const add = (a: Stats, b: Stats, s = 1): Stats => ({
  n: a.n + s * b.n, sy: a.sy + s * b.sy, yy: a.yy + s * b.yy,
  sx: a.sx.map((v, j) => v + s * b.sx[j]), xy: a.xy.map((v, j) => v + s * b.xy[j]),
  xx: a.xx.map((r, i) => r.map((v, j) => v + s * b.xx[i][j])),
});
const zero = (): Stats => ({ n: 0, sy: 0, yy: 0, sx: Array(P).fill(0), xy: Array(P).fill(0), xx: Array.from({ length: P }, () => Array(P).fill(0)) });
const TOTAL = FOLDS.reduce((a, f) => add(a, f as Stats), zero());
const TRAIN = FOLDS.map((f) => add(TOTAL, f as Stats, -1));

/** 중심화한 Gram 행렬과 Xᵀy */
function centered(s: Stats) {
  const mx = s.sx.map((v) => v / s.n), my = s.sy / s.n;
  const G = s.xx.map((r, i) => r.map((v, j) => v - s.n * mx[i] * mx[j]));
  const c = s.xy.map((v, j) => v - s.n * mx[j] * my);
  return { G, c, mx, my, n: s.n };
}
type Cent = ReturnType<typeof centered>;
const CT = centered(TOTAL), CTR = TRAIN.map(centered);

/** Lasso 좌표 하강법(warm start w0) */
export function lasso(cs: Cent, a: number, w0?: number[]) {
  const w = w0 ? [...w0] : Array(P).fill(0);
  const { G, c, n } = cs;
  for (let it = 0; it < 20000; it++) {
    let dmax = 0;
    for (let j = 0; j < P; j++) {
      let r = c[j];
      for (let k = 0; k < P; k++) if (k !== j) r -= G[j][k] * w[k];
      r /= n;
      const nw = Math.sign(r) * Math.max(Math.abs(r) - a, 0) / (G[j][j] / n);
      dmax = Math.max(dmax, Math.abs(nw - w[j]));
      w[j] = nw;
    }
    if (dmax < 1e-10) break;
  }
  return w;
}
/** Ridge: (G + λI) w = c, λ = 2nα */
export function ridge(cs: Cent, a: number) {
  const lam = 2 * cs.n * a;
  const A = cs.G.map((r, i) => [...r.map((v, j) => v + (i === j ? lam : 0)), cs.c[i]]);
  for (let k = 0; k < P; k++) {
    let p = k;
    for (let i = k + 1; i < P; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
    [A[k], A[p]] = [A[p], A[k]];
    for (let i = k + 1; i < P; i++) { const m = A[i][k] / A[k][k]; for (let j = k; j <= P; j++) A[i][j] -= m * A[k][j]; }
  }
  const w = Array(P).fill(0);
  for (let i = P - 1; i >= 0; i--) { let s = A[i][P]; for (let j = i + 1; j < P; j++) s -= A[i][j] * w[j]; w[i] = s / A[i][i]; }
  return w;
}
/** 검증 fold의 평균 제곱오차(절편 b = ȳ_tr − x̄_trᵀw) */
function valMSE(f: Fold, cs: Cent, w: number[]) {
  const b = cs.my - cs.mx.reduce((s, v, j) => s + v * w[j], 0);
  let wxy = 0, wsx = 0, wGw = 0;
  for (let i = 0; i < P; i++) { wxy += w[i] * f.xy[i]; wsx += w[i] * f.sx[i]; for (let j = 0; j < P; j++) wGw += w[i] * f.xx[i][j] * w[j]; }
  return (f.yy - 2 * b * f.sy - 2 * wxy + f.n * b * b + 2 * b * wsx + wGw) / f.n;
}

type Kind = "lasso" | "ridge";
export const A_MAX = Math.max(...CT.c.map((v) => Math.abs(v))) / CT.n;
export const GRID = Array.from({ length: 100 }, (_, i) => A_MAX * Math.pow(10, (-3 * i) / 99));

/** 격자 전체의 계수 경로와 CV 오차 */
export function path(kind: Kind) {
  const coef: number[][] = [], cv: number[] = [];
  let w = Array(P).fill(0);
  const wf = CTR.map(() => Array(P).fill(0));
  for (const a of GRID) {
    w = kind === "lasso" ? lasso(CT, a, w) : ridge(CT, a);
    coef.push(w);
    let s = 0;
    CTR.forEach((cs, k) => { wf[k] = kind === "lasso" ? lasso(cs, a, wf[k]) : ridge(cs, a); s += valMSE(FOLDS[k], cs, wf[k]); });
    cv.push(s / FOLDS.length);
  }
  return { coef, cv };
}
/** 임의의 α에서의 계수와 CV 오차 */
export function at(kind: Kind, a: number, warm?: number[]) {
  const w = kind === "lasso" ? lasso(CT, a, warm) : ridge(CT, a);
  const cv = CTR.reduce((s, cs, k) => s + valMSE(FOLDS[k], cs, kind === "lasso" ? lasso(cs, a) : ridge(cs, a)), 0) / FOLDS.length;
  return { w, cv };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "정규화 경로", "manipulate", 0.58, "강도 α에 따른 특징 10개의 계수 경로와 5-fold 교차검증 오차 곡선");
  let kind: Kind = "lasso", la = 1.0; // log10 α
  const paths: Record<Kind, ReturnType<typeof path>> = { lasso: path("lasso"), ridge: path("ridge") };
  let cur = at(kind, 10 ** la);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 40, PR = 38, top = 18, split = Math.round(H * 0.6), bottom = H - 22;
    const l0 = Math.log10(GRID[0]), l1 = Math.log10(GRID[GRID.length - 1]);
    const X = (lg: number) => PL + ((lg - l1) / (l0 - l1)) * (W - PL - PR); // 오른쪽 = 강한 정규화(슬라이더와 같은 방향)
    const { coef, cv } = paths[kind];
    const all = coef.flat();
    const cmax = Math.max(...all.map(Math.abs)) * 1.08;
    const Y = (v: number) => top + ((cmax - v) / (2 * cmax)) * (split - 14 - top);
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, Y(0)); ctx.lineTo(W - PR, Y(0)); ctx.stroke();
    ctx.fillStyle = C.mid;
    for (const v of [-cmax * 0.8, 0, cmax * 0.8]) { const r = Math.round(v / 10) * 10; ctx.fillText(r.toFixed(0), W - PR + 6, Y(r) + 4); }
    ctx.fillText("계수", W - PR + 6, top - 6);
    // 경로
    const xs = GRID.map((a) => X(Math.log10(a)));
    for (let j = 0; j < P; j++) {
      const on = Math.abs(cur.w[j]) > 1e-9;
      ctx.strokeStyle = on ? C.edge : alpha(C.mid, 0.55); ctx.lineWidth = on ? 1.8 : 1.2;
      ctx.beginPath(); coef.forEach((w, i) => (i ? ctx.lineTo(xs[i], Y(w[j])) : ctx.moveTo(xs[i], Y(w[j])))); ctx.stroke();
    }
    // 왼쪽 끝(가장 약한 정규화) 이름표(겹치지 않게 밀어낸다)
    const last = coef[coef.length - 1];
    const tags = NAMES.map((nm, j) => ({ nm, j, y: Y(last[j]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < tags.length; i++) tags[i].y = Math.max(tags[i].y, tags[i - 1].y + 10);
    const over = tags[tags.length - 1].y - (split - 14);
    if (over > 0) tags.forEach((t) => (t.y -= over));
    ctx.font = "10px JetBrains Mono, monospace";
    for (const t of tags) { ctx.fillStyle = Math.abs(cur.w[t.j]) > 1e-9 ? C.edge : C.mid; ctx.fillText(t.nm, PL - 6 - ctx.measureText(t.nm).width, t.y + 3); }
    // 아래: CV 오차
    const lo = Math.min(...cv), hi = Math.min(Math.max(...cv), lo * 2.2);
    const Y2 = (v: number) => bottom - ((Math.min(v, hi) - lo) / (hi - lo)) * (bottom - split - 16);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, bottom); ctx.lineTo(W - PR, bottom); ctx.stroke();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); cv.forEach((v, i) => (i ? ctx.lineTo(xs[i], Y2(v)) : ctx.moveTo(xs[i], Y2(v)))); ctx.stroke();
    const best = cv.indexOf(lo);
    ctx.fillStyle = C.good; ctx.beginPath(); ctx.arc(xs[best], Y2(lo), 4.5, 0, 7); ctx.fill();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("5-fold CV MSE", PL + 4, split + 4);
    ctx.fillText(hi.toFixed(0), W - PR + 6, Y2(hi) + 8);
    ctx.fillText(lo.toFixed(0), W - PR + 6, bottom - 4);
    // α 눈금
    for (const a of [0.1, 1, 10]) { const x = X(Math.log10(a)), s = String(a); ctx.fillText(s, x - ctx.measureText(s).width / 2, bottom + 14); }
    ctx.fillText("α", W - PR + 6, bottom + 14);
    // 현재 α
    const cx = X(la);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(cx, Y2(cur.cv), 4.5, 0, 7); ctx.fill();
  }

  let raf = 0;
  function update() {
    const a = 10 ** la;
    cur = at(kind, a);
    const nz = cur.w.map((v, j) => (Math.abs(v) > 1e-9 ? NAMES[j] : "")).filter(Boolean);
    const { cv } = paths[kind];
    const lo = Math.min(...cv), best = GRID[cv.indexOf(lo)];
    ro.set("a", kind === "lasso" ? `α = ${fmtA(a)}` : `α = ${fmtA(a)} (λ = 2Nα = ${fmtA(2 * TOTAL.n * a)})`);
    ro.set("nz", `0 아닌 계수 ${nz.length}/${P}개`, kind === "lasso" ? "hot" : "");
    ro.set("sel", nz.length === P ? "선택: 전부" : `선택: ${nz.join(", ") || "없음"}`);
    ro.set("cv", `CV MSE ${cur.cv.toFixed(0)}`);
    ro.set("best", `CV 최소 ${lo.toFixed(0)} (α = ${fmtA(best)})`, "ok");
    pageState.set("regularization-path", { kind, alpha: a, nonzero: nz.length, cv: cur.cv }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  const fmtA = (a: number) => (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a >= 1 ? a.toFixed(2) : a.toPrecision(2));

  const pre = presets<Kind>([{ key: "lasso", label: "Lasso (L1)" }, { key: "ridge", label: "Ridge (L2)" }], (k) => { kind = k; update(); }, kind);
  pre.row.classList.add("segmented");
  const pa = param({
    label: "α", min: -1.34, max: 1.65, step: 0.01, value: la, fmt: (v) => fmtA(10 ** v),
    aria: "정규화 강도 α (로그 눈금, 오른쪽으로 갈수록 강하다)", onInput: (v) => { la = v; update(); },
  });
  const grid = el("div", "param-grid");
  grid.append(pa.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
