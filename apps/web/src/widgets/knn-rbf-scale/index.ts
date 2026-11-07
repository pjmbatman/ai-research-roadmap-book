// kNN과 RBF 길이척도 (data/widgets.toml: knn-rbf-scale).
// 16.3장의 두 예를 그대로 쓴다.
//   1D 사인: 본문 셀의 측정 30점(np.random.default_rng(3), x ∈ [0, 10], y = sin x + N(0, 0.1²))을 그대로 담았다.
//            오차는 학습 범위 [0, 10]의 격자(0.1 간격)에서 참 함수 sin x와 비교한 RMSE다.
//   2D 농도·온도: 농도 0.01–0.2 mol/L, 온도 280–360 K, 속도 = 50c²·exp(−2000(1/T − 1/320)) + N(0, 0.02²)
//            (본문과 같은 식, 이 위젯의 고정 시드로 600개를 만들고 400개 학습 / 200개 시험). 오차는 시험 R².
// kNN 회귀: 가까운 k개의 평균. RBF 커널 릿지: f(x) = k(x)ᵀ(K + λI)⁻¹y, k = exp(−‖x − x′‖²/(2ℓ²)), λ = 잡음 분산.
// 표준화 스위치는 학습 데이터의 평균·표준편차로 두 특징을 맞춘다(2D에서만 의미가 있다).
import { frame, param, presets, readouts, el, C, rng, rgbOf } from "../frame";
import { pageState } from "../../client/store";

const X1 = [0.014901,0.303460,0.856492,0.941286,1.136720,1.597389,2.368105,2.842012,2.927207,2.984012,3.139860,3.912282,4.306280,4.331269,4.713097,4.790513,5.167402,5.821620,5.851629,5.867986,6.485472,6.962160,7.069651,7.345772,7.378378,7.732770,8.012745,8.917111,9.562673,9.734603];
const Y1 = [-0.073746,0.269652,0.843803,0.866351,0.916411,1.066657,0.415819,0.397251,0.116782,-0.009933,0.029377,-0.626575,-0.963142,-1.035889,-0.997387,-1.002225,-0.757696,-0.370609,-0.398903,-0.292209,0.180358,0.535405,0.766267,0.931871,0.867534,0.914382,1.010339,0.236751,-0.068446,-0.255755];
const XG1 = Array.from({ length: 141 }, (_, i) => -2 + 0.1 * i);
const IN1 = XG1.map((_, i) => i >= 20 && i <= 120);

export const rate = (c: number, t: number) => 50 * c * c * Math.exp(-2000 * (1 / t - 1 / 320));
export const D2 = (() => {
  const r = rng(1603);
  const pts = Array.from({ length: 600 }, () => { const c = 0.01 + 0.19 * r.u(), t = 280 + 80 * r.u(); return { c, t }; });
  const y = pts.map(({ c, t }) => rate(c, t) + 0.02 * r.normal());
  const X = pts.map(({ c, t }) => [c, t]);
  return { Xtr: X.slice(0, 400), ytr: y.slice(0, 400), Xte: X.slice(400), yte: y.slice(400) };
})();
const GC = 40, GT = 28; // 2D 그림 격자

type Model = "knn" | "rbf";
type Data = "d1" | "d2";

function knnPredict(Xtr: number[][], ytr: number[], Q: number[][], k: number) {
  const n = Xtr.length, d = new Float64Array(n), idx = new Int32Array(n);
  return Q.map((q) => {
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < q.length; j++) s += (Xtr[i][j] - q[j]) ** 2; d[i] = s; idx[i] = i; }
    // 부분 선택 정렬: 가장 가까운 k개
    const kk = Math.min(k, n);
    for (let a = 0; a < kk; a++) { let b = a; for (let i = a + 1; i < n; i++) if (d[idx[i]] < d[idx[b]]) b = i; const t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    let s = 0; for (let a = 0; a < kk; a++) s += ytr[idx[a]];
    return s / kk;
  });
}
/** K + λI의 Cholesky로 α = (K + λI)⁻¹y를 구한다 */
function krrFit(Xtr: number[][], ytr: number[], ell: number, lam: number) {
  const n = Xtr.length, g = 1 / (2 * ell * ell);
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = 0; for (let m = 0; m < Xtr[i].length; m++) s += (Xtr[i][m] - Xtr[j][m]) ** 2; L[i * n + j] = Math.exp(-g * s) + (i === j ? lam : 0); }
  for (let j = 0; j < n; j++) {
    let s = L[j * n + j];
    for (let m = 0; m < j; m++) s -= L[j * n + m] ** 2;
    const dj = Math.sqrt(Math.max(s, 1e-300));
    L[j * n + j] = dj;
    for (let i = j + 1; i < n; i++) { let t = L[i * n + j]; for (let m = 0; m < j; m++) t -= L[i * n + m] * L[j * n + m]; L[i * n + j] = t / dj; }
  }
  const z = new Float64Array(n), a = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = ytr[i]; for (let m = 0; m < i; m++) s -= L[i * n + m] * z[m]; z[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let m = i + 1; m < n; m++) s -= L[m * n + i] * a[m]; a[i] = s / L[i * n + i]; }
  return (Q: number[][]) => Q.map((q) => { let s = 0; for (let i = 0; i < n; i++) { let d = 0; for (let m = 0; m < q.length; m++) d += (Xtr[i][m] - q[m]) ** 2; s += a[i] * Math.exp(-g * d); } return s; });
}
const r2 = (p: number[], y: number[]) => { const m = y.reduce((s, v) => s + v, 0) / y.length; return 1 - p.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / y.reduce((s, v) => s + (v - m) ** 2, 0); };

/** 한 설정의 예측과 오차 */
export function evaluate(model: Model, data: Data, k: number, ell: number, std: boolean) {
  if (data === "d1") {
    const X = X1.map((x) => [x]), Q = XG1.map((x) => [x]);
    const pred = model === "knn" ? (q: number[][]) => knnPredict(X, Y1, q, k) : krrFit(X, Y1, ell, 0.01);
    const curve = pred(Q), fit = pred(X);
    const rin = Math.sqrt(curve.reduce((s, v, i) => s + (IN1[i] ? (v - Math.sin(XG1[i])) ** 2 : 0), 0) / IN1.filter(Boolean).length);
    const rtr = Math.sqrt(fit.reduce((s, v, i) => s + (v - Y1[i]) ** 2, 0) / Y1.length);
    return { kind: "d1" as const, curve, rin, rtr };
  }
  const mu = [0, 1].map((j) => D2.Xtr.reduce((s, x) => s + x[j], 0) / 400);
  const sd = [0, 1].map((j) => Math.sqrt(D2.Xtr.reduce((s, x) => s + (x[j] - mu[j]) ** 2, 0) / 400));
  const tf = (x: number[]) => (std ? x.map((v, j) => (v - mu[j]) / sd[j]) : x);
  const Xtr = D2.Xtr.map(tf), Xte = D2.Xte.map(tf);
  const pred = model === "knn" ? (q: number[][]) => knnPredict(Xtr, D2.ytr, q, k) : krrFit(Xtr, D2.ytr, ell, 0.02 ** 2);
  const grid: number[][] = [];
  for (let b = 0; b < GT; b++) for (let a = 0; a < GC; a++) grid.push(tf([0.01 + (0.19 * (a + 0.5)) / GC, 360 - (80 * (b + 0.5)) / GT]));
  const heat = pred(grid);
  const te = r2(pred(Xte), D2.yte), tr = r2(pred(Xtr), D2.ytr);
  // 거리 제곱의 기대값 중 온도가 차지하는 몫: E[(Δx_j)²] = 2·Var(x_j)
  const v = std ? [1, 1] : [sd[0] ** 2, sd[1] ** 2];
  return { kind: "d2" as const, heat, te, tr, tshare: v[1] / (v[0] + v[1]) };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "kNN과 RBF 길이척도", "manipulate", 0.5, "kNN 또는 RBF 커널 릿지 회귀의 적합: 1D에서는 측정점·참 함수·예측 곡선, 2D에서는 농도·온도 평면의 참 반응 속도와 예측 색지도");
  let model: Model = "knn", data: Data = "d2", k = 10, lell = 0, std = false;
  let res = evaluate(model, data, k, 10 ** lell, std);

  function shade(t: number) {
    const a = rgbOf(C.bg), b = rgbOf(C.edge), u = Math.max(0, Math.min(1, t));
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(",")})`;
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    if (res.kind === "d1") {
      const PL = 30, PR = 10, top = 14, bottom = H - 22;
      const X = (x: number) => PL + ((x + 2) / 14) * (W - PL - PR);
      const Y = (y: number) => top + ((1.6 - y) / 3.2) * (bottom - top);
      ctx.fillStyle = C.faint;
      ctx.globalAlpha = 0.35; ctx.fillRect(PL, top, X(0) - PL, bottom - top); ctx.fillRect(X(10), top, W - PR - X(10), bottom - top); ctx.globalAlpha = 1;
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(PL, Y(0)); ctx.lineTo(W - PR, Y(0)); ctx.stroke();
      ctx.fillStyle = C.mid;
      for (const v of [-1, 0, 1]) ctx.fillText(String(v), PL - 8 - ctx.measureText(String(v)).width + 4, Y(v) + 4);
      for (let x = 0; x <= 10; x += 2) ctx.fillText(String(x), X(x) - 3, bottom + 14);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      ctx.beginPath(); XG1.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(Math.sin(x))) : ctx.moveTo(X(x), Y(Math.sin(x))))); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
      ctx.beginPath(); XG1.forEach((x, i) => { const y = Math.max(-1.6, Math.min(1.6, res.kind === "d1" ? res.curve[i] : 0)); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }); ctx.stroke();
      ctx.fillStyle = C.ink2;
      X1.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(Y1[i]), 3.2, 0, 7); ctx.fill(); });
      ctx.fillStyle = C.mid; ctx.fillText("x", W - PR - 8, bottom + 14);
      return;
    }
    // 2D: 왼쪽 참 함수, 오른쪽 예측
    const gap = 14, top = 18, bottom = H - 30, PL = 30;
    const pw = (W - PL - gap - 8) / 2, ph = bottom - top;
    const vmax = rate(0.2, 360);
    const truth: number[] = [];
    for (let b = 0; b < GT; b++) for (let a = 0; a < GC; a++) truth.push(rate(0.01 + (0.19 * (a + 0.5)) / GC, 360 - (80 * (b + 0.5)) / GT));
    const panels: [string, number[], number][] = [["참 반응 속도", truth, PL], [model === "knn" ? `kNN 예측 (k = ${k})` : `RBF 예측 (ℓ = ${fmtL(10 ** lell)})`, res.heat, PL + pw + gap]];
    for (const [name, vals, x0] of panels) {
      const cw = pw / GC, chh = ph / GT;
      for (let b = 0; b < GT; b++) for (let a = 0; a < GC; a++) { ctx.fillStyle = shade(vals[b * GC + a] / vmax); ctx.fillRect(x0 + a * cw, top + b * chh, Math.ceil(cw), Math.ceil(chh)); }
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, top, pw, ph);
      ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
      ctx.fillText(name, x0, top - 5);
      ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
      ctx.fillText("0.01", x0, bottom + 13);
      const s = "0.2 mol/L"; ctx.fillText(s, x0 + pw - ctx.measureText(s).width, bottom + 13);
    }
    ctx.fillStyle = C.mid;
    ctx.save(); ctx.translate(11, (top + bottom) / 2); ctx.rotate(-Math.PI / 2); const tl = "온도 280–360 K"; ctx.fillText(tl, -ctx.measureText(tl).width / 2, 0); ctx.restore();
    ctx.fillText("360", PL - 26, top + 9); ctx.fillText("280", PL - 26, bottom);
  }
  const fmtL = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

  let raf = 0;
  function update() {
    kRow.row.hidden = model !== "knn";
    lRow.row.hidden = model !== "rbf";
    tools.hidden = data !== "d2";
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      res = evaluate(model, data, k, 10 ** lell, std);
      if (res.kind === "d1") {
        ro.set("a", `범위 안 RMSE ${res.rin.toFixed(3)} (참 함수 대비)`, res.rin < 0.12 ? "ok" : res.rin > 0.3 ? "hot" : "");
        ro.set("b", `학습 RMSE ${res.rtr.toFixed(3)}`);
        ro.set("c", "측정 잡음 σ = 0.1");
      } else {
        ro.set("a", `시험 R² ${res.te.toFixed(3)}`, res.te > 0.95 ? "ok" : res.te < 0.6 ? "hot" : "");
        ro.set("b", `학습 R² ${res.tr.toFixed(3)}`);
        ro.set("c", `거리² 중 온도 몫 ${(100 * res.tshare).toFixed(res.tshare > 0.999 ? 4 : 0)}%`, res.tshare > 0.9 ? "hot" : "");
      }
      pageState.set("knn-rbf-scale", { model, data, k, ell: 10 ** lell, std }, root);
      draw();
    });
  }

  const pm = presets<Model>([{ key: "knn", label: "kNN" }, { key: "rbf", label: "RBF 커널 릿지" }], (v) => { model = v; update(); }, model);
  pm.row.classList.add("segmented");
  const pd = presets<Data>([{ key: "d1", label: "1D 사인" }, { key: "d2", label: "2D 농도·온도" }], (v) => { data = v; update(); }, data);
  const kRow = param({ label: "k", min: 1, max: 30, step: 1, value: k, fmt: (v) => String(v), aria: "이웃 수 k", onInput: (v) => { k = v; update(); } });
  const lRow = param({ label: "ℓ", min: -1.5, max: 1.3, step: 0.02, value: lell, fmt: (v) => fmtL(10 ** v), aria: "RBF 길이척도 ℓ (로그 눈금)", onInput: (v) => { lell = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(kRow.row, lRow.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 특징 표준화`;
  sw.querySelector("input")!.addEventListener("change", (e) => { std = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(pm.row, pd.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
