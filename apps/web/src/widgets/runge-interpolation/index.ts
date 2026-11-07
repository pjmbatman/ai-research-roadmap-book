// Runge 현상 (data/widgets.toml: runge-interpolation).
// f(x) = 1/(1 + 25x²)를 [−1, 1]에서 노드 n+1개로 보간한다. 다항식은 무게중심(barycentric) 공식으로,
// cubic spline은 not-a-knot 끝 조건(scipy CubicSpline 기본값)으로 계산한다. 최대 오차는 2001개 시험점에서 잰다(9.5장 셀과 같다).
// 위: f와 보간 함수·노드. 아래: 오차 |f − p|의 로그 눈금. 조작: n 슬라이더와 방법 스위치.
import { frame, param, presets, readouts, el, C } from "../frame";
import { pageState } from "../../client/store";

export const runge = (x: number) => 1 / (1 + 25 * x * x);
const NT = 2001;
const XT = Array.from({ length: NT }, (_, i) => -1 + (2 * i) / (NT - 1));
export type Method = "equi" | "cheb" | "spline";
const NAMES: Record<Method, string> = { equi: "등간격 다항식", cheb: "Chebyshev 다항식", spline: "cubic spline" };

export const equiNodes = (n: number) => Array.from({ length: n + 1 }, (_, j) => -1 + (2 * j) / n);
export const chebNodes = (n: number) => Array.from({ length: n + 1 }, (_, j) => Math.cos((Math.PI * (2 * j + 1)) / (2 * n + 2)));

/** 무게중심 보간(2형): p(x) = Σ w_j y_j/(x − x_j) / Σ w_j/(x − x_j), w_j = 1/Π_{k≠j}(x_j − x_k) */
export function barycentric(xn: number[], yn: number[]) {
  const m = xn.length;
  const w = xn.map((xj, j) => { let p = 1; for (let k = 0; k < m; k++) if (k !== j) p *= xj - xn[k]; return 1 / p; });
  const s = Math.max(...w.map(Math.abs));
  for (let j = 0; j < m; j++) w[j] /= s;
  return (x: number) => {
    let num = 0, den = 0;
    for (let j = 0; j < m; j++) {
      const d = x - xn[j];
      if (d === 0) return yn[j];
      const t = w[j] / d;
      num += t * yn[j]; den += t;
    }
    return num / den;
  };
}

/** 작은 밀집 선형계 A x = b (부분 피벗 Gauss 소거) */
function solveDense(A: number[][], b: number[]) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j]; x[r] = s / M[r][r]; }
  return x;
}

/** not-a-knot cubic spline(노드 4개 이상). 노드의 2계 도함수 M_i를 풀고 구간마다 3차식으로 계산한다. */
export function cubicSpline(xn: number[], yn: number[]) {
  const n = xn.length - 1;
  const h = Array.from({ length: n }, (_, i) => xn[i + 1] - xn[i]);
  const A = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0));
  const b = new Array(n + 1).fill(0);
  // 끝 조건: 첫 두 구간과 마지막 두 구간의 3계 도함수가 같다(not-a-knot)
  A[0][0] = -1 / h[0]; A[0][1] = 1 / h[0] + 1 / h[1]; A[0][2] = -1 / h[1];
  A[n][n - 2] = -1 / h[n - 2]; A[n][n - 1] = 1 / h[n - 2] + 1 / h[n - 1]; A[n][n] = -1 / h[n - 1];
  for (let i = 1; i < n; i++) {
    A[i][i - 1] = h[i - 1]; A[i][i] = 2 * (h[i - 1] + h[i]); A[i][i + 1] = h[i];
    b[i] = 6 * ((yn[i + 1] - yn[i]) / h[i] - (yn[i] - yn[i - 1]) / h[i - 1]);
  }
  const M = solveDense(A, b);
  return (x: number) => {
    let i = 0;
    while (i < n - 1 && x > xn[i + 1]) i++;
    const a = xn[i + 1] - x, c = x - xn[i], hi = h[i];
    return (M[i] * a ** 3 + M[i + 1] * c ** 3) / (6 * hi) + (yn[i] / hi - (M[i] * hi) / 6) * a + (yn[i + 1] / hi - (M[i + 1] * hi) / 6) * c;
  };
}

export function interpolant(m: Method, n: number) {
  const xn = m === "cheb" ? chebNodes(n) : equiNodes(n);
  const yn = xn.map(runge);
  return { xn, yn, p: m === "spline" ? cubicSpline(xn, yn) : barycentric(xn, yn) };
}
export function maxError(m: Method, n: number) {
  const { p } = interpolant(m, n);
  let e = 0;
  for (const x of XT) e = Math.max(e, Math.abs(p(x) - runge(x)));
  return e;
}

const sci = (v: number) => (v >= 0.01 && v < 100 ? v.toFixed(v < 1 ? 3 : 1) : v.toExponential(1).replace("e+", "e").replace("e-", "e−"));

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "Runge 현상", "manipulate", narrow ? 0.85 : 0.6, "f(x) = 1/(1 + 25x²)와 보간 함수, 노드, 그리고 로그 눈금의 오차 곡선");
  let n = 10, method: Method = "equi";
  let curve = new Float64Array(0), err = new Float64Array(0), nodes: number[] = [];
  const Y0 = -0.6, Y1 = 1.4, E0 = -6, E1 = 3;

  function compute() {
    const it = interpolant(method, n);
    nodes = it.xn;
    curve = new Float64Array(NT); err = new Float64Array(NT);
    for (let i = 0; i < NT; i++) { curve[i] = it.p(XT[i]); err[i] = Math.abs(curve[i] - runge(XT[i])); }
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 34, R = 12, T = 12, gap = 26, split = Math.round((H - T - 22 - gap) * 0.66);
    const top = { y: T, h: split }, bot = { y: T + split + gap, h: H - T - split - gap - 22 };
    const X = (x: number) => L + ((x + 1) / 2) * (W - L - R);
    const Y = (y: number) => top.y + ((Y1 - y) / (Y1 - Y0)) * top.h;
    const YE = (e: number) => bot.y + ((E1 - Math.log10(Math.max(e, 1e-12))) / (E1 - E0)) * bot.h;
    ctx.font = "11px JetBrains Mono, monospace";
    // 축과 격자
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    ctx.strokeRect(L, top.y, W - L - R, top.h);
    ctx.strokeRect(L, bot.y, W - L - R, bot.h);
    ctx.textAlign = "right";
    for (const v of [0, 1]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillText(String(v), L - 5, Y(v) + 4); }
    for (const k of [-4, -2, 0, 2]) {
      ctx.beginPath(); ctx.moveTo(L, YE(10 ** k)); ctx.lineTo(W - R, YE(10 ** k)); ctx.stroke();
      ctx.fillText(k === 0 ? "1" : `1e${k}`.replace("-", "−"), L - 5, YE(10 ** k) + 4);
    }
    ctx.textAlign = "center";
    for (const v of [-1, -0.5, 0, 0.5, 1]) ctx.fillText(String(v).replace("-", "−"), X(v), H - 6);
    ctx.textAlign = "start";
    ctx.fillText("|f − p|", L + 6, bot.y + 13);
    // 위: f(정답, 굵은 흐린 선)와 보간 함수
    ctx.save(); ctx.beginPath(); ctx.rect(L, top.y, W - L - R, top.h); ctx.clip();
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.lineJoin = "round";
    ctx.beginPath(); XT.forEach((x, i) => (i % 4 === 0 ? (i ? ctx.lineTo(X(x), Y(runge(x))) : ctx.moveTo(X(x), Y(runge(x)))) : 0)); ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < NT; i++) { const y = Math.max(Y0 - 1, Math.min(Y1 + 1, curve[i])); i ? ctx.lineTo(X(XT[i]), Y(y)) : ctx.moveTo(X(XT[i]), Y(y)); }
    ctx.stroke();
    ctx.fillStyle = C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5;
    for (const x of nodes) { ctx.beginPath(); ctx.arc(X(x), Y(runge(x)), 3.6, 0, 7); ctx.fill(); ctx.stroke(); }
    ctx.restore();
    // 범례
    ctx.font = "11px JetBrains Mono, monospace";
    const lx = X(-0.72), ly = top.y + 16;
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(lx, ly - 4); ctx.lineTo(lx + 16, ly - 4); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.fillText("f(x)", lx + 22, ly);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(lx, ly + 12); ctx.lineTo(lx + 16, ly + 12); ctx.stroke();
    ctx.fillText("p(x)", lx + 22, ly + 16);
    // 아래: 오차
    ctx.save(); ctx.beginPath(); ctx.rect(L, bot.y, W - L - R, bot.h); ctx.clip();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < NT; i++) { const y = Math.min(YE(err[i]), bot.y + bot.h + 2); i ? ctx.lineTo(X(XT[i]), y) : ctx.moveTo(X(XT[i]), y); }
    ctx.stroke();
    ctx.restore();
  }

  const all: Record<Method, number> = { equi: 0, cheb: 0, spline: 0 };
  function update() {
    compute();
    (Object.keys(NAMES) as Method[]).forEach((m) => (all[m] = maxError(m, n)));
    ro.set("nodes", `노드 ${n + 1}개`);
    (Object.keys(NAMES) as Method[]).forEach((m) => ro.set(m, `${NAMES[m]} 최대 오차 ${sci(all[m])}`, m === method ? (all[m] > 0.5 ? "hot" : "ok") : ""));
    pageState.set("runge", { n, method }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const mb = presets((Object.keys(NAMES) as Method[]).map((k) => ({ key: k, label: k === "equi" ? "등간격" : k === "cheb" ? "Chebyshev" : "cubic spline" })), (k) => { method = k; update(); }, method);
  mb.row.classList.add("segmented");
  const pn = param({ label: "n", min: 4, max: 30, step: 1, value: n, fmt: (v) => String(v), aria: "보간 차수 n(노드 n+1개)", onInput: (v) => { n = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const ro = readouts();
  f.panel.append(mb.row, grid, ro.row);
  f.onResize(draw);
  update();
}
