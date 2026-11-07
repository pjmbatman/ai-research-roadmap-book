// Tikhonov L-curve (data/widgets.toml: l-curve). 24.7장 역문제 절의 예제 그대로다.
// 80개 격자 위 신호(상자 + 가우시안 봉우리)를 폭 0.03의 가우시안 커널(행 정규화)로 흐리게 하고 잡음 σ = 0.01을 더했다.
// 잡음 벡터는 장의 코드(np.random.default_rng(0))가 만든 값을 그대로 넣어, 장의 출력과 같은 숫자가 나온다.
// x_λ = (AᵀA + λI)⁻¹Aᵀy를 AᵀA의 고윳값 분해(Jacobi)로 모든 λ에 대해 한 번에 계산한다.
// 왼쪽(좁은 화면에서는 위): (log 잔차, log 해 노름) 곡선과 현재 λ, 모서리(곡률 최대), 불일치 원리 선. 오른쪽(아래): 참 신호·측정·복원.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export const N = 80, SIGMA = 0.01;
const NOISE = [0.1257302, -0.1321049, 0.6404227, 0.1049001, -0.5356694, 0.3615951, 1.304, 0.947081, -0.7037352, -1.2654215, -0.6232745, 0.041326, -2.3250308, -0.2187917, -1.2459109, -0.7322674, -0.544259, -0.3163002, 0.4116305, 1.0425134, -0.1285347, 1.3664635, -0.6651947, 0.3515101, 0.9034702, 0.0940123, -0.7434992, -0.9217254, -0.4577258, 0.2201951, -1.0096182, -0.2091756, -0.159225, 0.5408456, 0.2146591, 0.3553727, -0.6538286, -0.1296136, 0.7839755, 1.4934311, -1.2590655, 1.5139238, 1.3458754, 0.7813114, 0.2644556, -0.3139228, 1.4580207, 1.9602583, 1.8016349, 1.3151038, 0.3573804, -1.2083186, -0.0044541, 0.6564749, -1.2883615, 0.3951221, 0.4298637, 0.6960427, -1.184118, -0.6617026, -0.4364352, -1.1698019, 1.7393679, -0.4959107, 0.3289696, -0.2585725, 1.5834729, 1.320361, 0.6333526, -2.2035099, 0.052029, 0.6836862, 1.0039616, -0.617907, 1.8220114, -1.320431, -0.661528, 0.93505, 0.0490546, 2.0023926];
/** λ 격자: 장과 같은 np.logspace(-8, 1, 91) */
export const LOGLAMS = Array.from({ length: 91 }, (_, i) => -8 + i * 0.1);

/** 대칭 행렬의 고윳값 분해(cyclic Jacobi). 행 우선 n×n. V의 열이 고유벡터. */
export function jacobiEig(S: Float64Array, n: number) {
  const a = Float64Array.from(S), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {   // 열 p, q
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {   // 행 p, q
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  return { w: Array.from({ length: n }, (_, i) => a[i * n + i]), V };
}

export function setup() {
  const s = Array.from({ length: N }, (_, i) => i / (N - 1));
  const A = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    let rs = 0;
    for (let j = 0; j < N; j++) { const v = Math.exp(-0.5 * (s[i] - s[j]) ** 2 / 0.03 ** 2); A[i * N + j] = v; rs += v; }
    for (let j = 0; j < N; j++) A[i * N + j] /= rs;
  }
  const xTrue = s.map((u) => (Math.abs(u - 0.3) < 0.1 ? 1 : 0) + 0.5 * Math.exp(-0.5 * (u - 0.7) ** 2 / 0.05 ** 2));
  const y = s.map((_, i) => { let v = 0; for (let j = 0; j < N; j++) v += A[i * N + j] * xTrue[j]; return v + SIGMA * NOISE[i]; });
  const AtA = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
    let v = 0; for (let k = 0; k < N; k++) v += A[k * N + i] * A[k * N + j];
    AtA[i * N + j] = v; AtA[j * N + i] = v;
  }
  const { w, V } = jacobiEig(AtA, N);
  const Aty = s.map((_, j) => { let v = 0; for (let k = 0; k < N; k++) v += A[k * N + j] * y[k]; return v; });
  const coef = w.map((_, i) => { let v = 0; for (let k = 0; k < N; k++) v += V[k * N + i] * Aty[k]; return v; });
  /** x_λ = Σ_i v_i (v_iᵀAᵀy)/(w_i + λ) */
  const solve = (lam: number) => {
    const x = new Array(N).fill(0);
    for (let i = 0; i < N; i++) { const g = coef[i] / (w[i] + lam); for (let k = 0; k < N; k++) x[k] += V[k * N + i] * g; }
    return x;
  };
  const norm = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  const stats = (lam: number) => {
    const x = solve(lam);
    const r = y.map((yi, i) => { let v = -yi; for (let j = 0; j < N; j++) v += A[i * N + j] * x[j]; return v; });
    return { x, res: norm(r), sol: norm(x), err: norm(x.map((v, i) => v - xTrue[i])) };
  };
  const curve = LOGLAMS.map((l) => stats(10 ** l));
  // 곡률: 장과 같은 np.gradient 방식(가운데는 중심차분, 양 끝은 한쪽 차분)
  const grad = (v: number[]) => v.map((_, i) => (i === 0 ? v[1] - v[0] : i === v.length - 1 ? v[i] - v[i - 1] : (v[i + 1] - v[i - 1]) / 2));
  const r = curve.map((c) => Math.log(c.res)), e = curve.map((c) => Math.log(c.sol));
  const dr = grad(r), de = grad(e), ddr = grad(dr), dde = grad(de);
  const curv = dr.map((_, i) => (dr[i] * dde[i] - de[i] * ddr[i]) / (dr[i] ** 2 + de[i] ** 2) ** 1.5);
  let iCorner = 5; for (let i = 5; i < curv.length - 5; i++) if (curv[i] > curv[iCorner]) iCorner = i;
  const target = Math.sqrt(N) * SIGMA;
  let iDisc = 0; curve.forEach((c, i) => { if (Math.abs(c.res - target) < Math.abs(curve[iDisc].res - target)) iDisc = i; });
  let iBest = 0; curve.forEach((c, i) => { if (c.err < curve[iBest].err) iBest = i; });
  return { s, xTrue, y, w, curve, iCorner, iDisc, iBest, target };
}

const sci = (v: number) => { const e = Math.floor(Math.log10(v) + 1e-9); const m = v / 10 ** e; return `${m.toFixed(1)}e${e < 0 ? "−" : ""}${Math.abs(e)}`; };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "Tikhonov L-curve", "manipulate", narrow ? 1.1 : 0.5, "λ에 따른 잔차 노름과 해 노름의 L-curve, 그리고 참 신호와 측정, 복원한 신호");
  const D = setup();
  let idx = 20; // λ = 1e-6
  type Key = "small" | "corner" | "disc" | "big";
  const at: Record<Key, number> = { small: 20, corner: D.iCorner, disc: D.iDisc, big: 80 };

  const LR0 = Math.log10(0.04), LR1 = Math.log10(4), LS0 = Math.log10(1), LS1 = Math.log10(Math.max(...D.curve.map((c) => c.sol)) * 1.4);
  function boxes() {
    const W = f.width, H = f.height;
    if (!narrow && W >= 400) {
      const lw = Math.min(W * 0.42, H + 10);
      return { lc: { x: 40, y: 10, w: lw - 40, h: H - 50 }, sg: { x: lw + 34, y: 10, w: W - lw - 44, h: H - 36 } };
    }
    const h1 = Math.round((H - 64) * 0.5);
    return { lc: { x: 40, y: 10, w: W - 50, h: h1 }, sg: { x: 40, y: h1 + 60, w: W - 50, h: H - h1 - 86 } };
  }
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { lc, sg } = boxes();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
    // L-curve
    const LX = (r: number) => lc.x + ((Math.log10(r) - LR0) / (LR1 - LR0)) * lc.w;
    const LY = (s: number) => lc.y + ((LS1 - Math.log10(s)) / (LS1 - LS0)) * lc.h;
    ctx.strokeStyle = C.faint; ctx.strokeRect(lc.x, lc.y, lc.w, lc.h);
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const v of [0.1, 1]) ctx.fillText(String(v), LX(v), lc.y + lc.h + 13);
    ctx.textAlign = "end"; ctx.fillText("잔차 ‖Ax − y‖", lc.x + lc.w, lc.y + lc.h + 27);
    ctx.textAlign = "right";
    for (const v of [1, 10]) if (Math.log10(v) <= LS1) ctx.fillText(String(v), lc.x - 5, LY(v) + 4);
    ctx.save(); ctx.translate(lc.x - 24, lc.y + lc.h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("해 노름 ‖x‖", 0, 0); ctx.restore();
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(lc.x, lc.y, lc.w, lc.h); ctx.clip();
    // 불일치 원리: 잔차 = √n σ
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(LX(D.target), lc.y); ctx.lineTo(LX(D.target), lc.y + lc.h); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); D.curve.forEach((c, i) => (i ? ctx.lineTo(LX(c.res), LY(c.sol)) : ctx.moveTo(LX(c.res), LY(c.sol)))); ctx.stroke();
    const cc = D.curve[D.iCorner];
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(LX(cc.res), LY(cc.sol), 6, 0, 7); ctx.stroke();
    const cur = D.curve[idx];
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(LX(cur.res), LY(cur.sol), 5, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = C.mid;
    ctx.fillText("√n σ", LX(D.target) + 4, lc.y + lc.h - 6);
    ctx.fillStyle = C.ink;
    ctx.fillText("모서리", LX(cc.res) + 9, LY(cc.sol) - 8);
    // 신호
    const V0 = -0.6, V1 = 1.6;
    const SX = (u: number) => sg.x + u * sg.w;
    const SY = (v: number) => sg.y + ((V1 - v) / (V1 - V0)) * sg.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(sg.x, sg.y, sg.w, sg.h);
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 1]) { ctx.beginPath(); ctx.moveTo(sg.x, SY(v)); ctx.lineTo(sg.x + sg.w, SY(v)); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(String(v), sg.x - 5, SY(v) + 4); }
    ctx.textAlign = "center";
    for (const v of [0, 0.5, 1]) ctx.fillText(String(v), SX(v), sg.y + sg.h + 13);
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(sg.x, sg.y, sg.w, sg.h); ctx.clip();
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.lineJoin = "round";
    ctx.beginPath(); D.xTrue.forEach((v, i) => (i ? ctx.lineTo(SX(D.s[i]), SY(v)) : ctx.moveTo(SX(D.s[i]), SY(v)))); ctx.stroke();
    ctx.fillStyle = C.mid;
    D.y.forEach((v, i) => { ctx.beginPath(); ctx.arc(SX(D.s[i]), SY(v), 1.8, 0, 7); ctx.fill(); });
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); cur.x.forEach((v, i) => { const c = Math.max(V0 - 1, Math.min(V1 + 1, v)); i ? ctx.lineTo(SX(D.s[i]), SY(c)) : ctx.moveTo(SX(D.s[i]), SY(c)); }); ctx.stroke();
    ctx.restore();
    // 범례
    const items: [string, (x: number, y: number) => void][] = [
      ["참 신호", (x, y) => { ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); }],
      ["측정 y", (x, y) => { ctx.fillStyle = C.mid; ctx.beginPath(); ctx.arc(x + 7, y, 2.4, 0, 7); ctx.fill(); }],
      ["x_λ", (x, y) => { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); }],
    ];
    let lx = sg.x + 6;
    const wsum = items.reduce((s, [t]) => s + ctx.measureText(t).width + 26, 0);
    ctx.fillStyle = C.bg; ctx.fillRect(sg.x + 1, sg.y + 1, Math.min(wsum, sg.w - 2), 20);
    for (const [t, mark] of items) { mark(lx, sg.y + 10); ctx.fillStyle = C.ink; ctx.fillText(t, lx + 18, sg.y + 14); lx += ctx.measureText(t).width + 26; }
  }

  function update() {
    const c = D.curve[idx];
    ro.set("lam", `λ = ${sci(10 ** LOGLAMS[idx])}`);
    ro.set("res", `잔차 ${c.res.toFixed(3)}`, c.res < D.target * 0.9 ? "hot" : "");
    ro.set("sol", `해 노름 ${c.sol.toFixed(2)}`);
    ro.set("err", `오차 ${c.err.toFixed(2)} (최소 ${D.curve[D.iBest].err.toFixed(2)})`, c.err < D.curve[D.iBest].err * 1.15 ? "ok" : c.err > 3 * D.curve[D.iBest].err ? "hot" : "");
    draw();
  }
  const pl = param({ label: "λ", min: -8, max: 1, step: 0.1, value: LOGLAMS[idx], fmt: (v) => sci(10 ** v), aria: "정규화 세기 λ의 로그", onInput: (v) => { idx = Math.round((v + 8) * 10); pre.select(null); update(); } });
  const pre = presets<Key>([
    { key: "small", label: "λ = 1e−6" }, { key: "corner", label: "L-curve 모서리" }, { key: "disc", label: "불일치 원리" }, { key: "big", label: "λ = 1" },
  ], (k) => { idx = at[k]; pl.set(LOGLAMS[idx]); update(); }, "small");
  const grid = el("div", "param-grid");
  grid.append(pl.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
