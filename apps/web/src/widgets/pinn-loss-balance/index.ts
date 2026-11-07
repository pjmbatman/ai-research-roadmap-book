// PINN 손실 균형 (data/widgets.toml: pinn-loss-balance).
// 24.2장의 1차원 Poisson 문제 −u'' = f, u(0) = u(1) = 0, 정확해 u* = sin πx + 0.2 sin 4πx와 같은 random feature 모델을 쓴다.
// u(x) = tanh(w x + b)·c, 뉴런 40개. w, b는 본문 셀과 같이 numpy default_rng(0)의 normal(0, 8, 40), uniform(−8, 8, 40) 값을 그대로 옮겼다.
// 손실 = (1/N_r)Σ(−u''(x_i) − f(x_i))² + (λ_bc/2)(u(0)² + u(1)²)를 ‖A c − rhs‖²로 쌓고(본문 pinn_system),
// '최소제곱'은 정확한 최소점(특잇값 분해로 구한 최소 노름 해, numpy lstsq와 같은 rcond), '경사하강'은 학습률 1/λ_max(AᵀA)로 20000 step이다.
// 읽음값은 본문과 같다: 학습점 residual RMS, 새 점(1001개) residual RMS, 경계 위반, 최대 해 오차.
import { frame, param, presets, readouts, el, C } from "../frame";

const W8 = [1.0058417687471464, -1.0568389063304151, 5.1233812035462565, 0.83920093722431766, -4.2853549852888877, 2.8927604392758779, 10.432000361041098, 7.5766477050339374, -5.6298818864559408, -10.12337176836842, -4.9861957002988175, 0.33060783477794881, -18.600246197110675, -1.7503333114603659, -9.9672875780245214, -5.8581388376276129, -4.3540718628584791, -2.5304012509532363, 3.2930442909930626, 8.3401069555414207, -1.0282773035522741, 10.931707764397487, -5.3215573878929083, 2.8120805607441577, 7.2277614532144687, 0.75209838208699653, -5.9479939948304672, -7.3738030100673555, -3.6618066053387133, 1.7615609877603953, -8.0769454683098871, -1.6734045989737045, -1.2738000793158217, 4.3267646774864614, 1.7172729800507271, 2.8429816723193713, -5.2306288753467154, -1.0369090695421557, 6.271803760490636, 11.947449161766086];
const B8 = [1.144477291676175, -2.8500897427849257, 1.5088004831951487, -2.5934203918858678, -1.7340959915494203, 6.2443896320766772, -4.3654785034659245, 1.9709943149766787, -6.6557545026818428, 5.3223063624543645, 4.5935729198189339, -4.1700889121127656, 6.023747692971261, -7.0629114431168905, -2.622127031269434, -5.595528529682575, -0.79457013361140838, 4.7411883245967079, -4.3097246561000411, -7.1676591829694463, -1.5271705628555488, -4.8237912878519147, -6.5479512700940496, 1.2853181757896106, -3.2208618748972384, 2.7519180473017499, -4.8077528965085872, 7.0738097681039651, -2.1582373080827431, -6.3120755268763276, 2.0657304246353476, 6.8344728490858788, -0.95396552454745631, 7.2734478990517957, -0.0016669809976477268, -1.1963420024147915, 1.9234152322460449, 7.9215440837651858, 7.1830987990042452, -0.63927777105446282];
const M = 40;
const PI = Math.PI;
const uTrue = (x: number) => Math.sin(PI * x) + 0.2 * Math.sin(4 * PI * x);
const fSrc = (x: number) => PI ** 2 * Math.sin(PI * x) + 0.2 * (4 * PI) ** 2 * Math.sin(4 * PI * x);
const phi = (x: number) => W8.map((w, j) => Math.tanh(w * x + B8[j]));
const phiXX = (x: number) => W8.map((w, j) => { const t = Math.tanh(w * x + B8[j]); return -2 * t * (1 - t * t) * w * w; });
const NT = 1001;
const XT = Array.from({ length: NT }, (_, i) => i / (NT - 1));
const PT = XT.map(phi), PXT = XT.map(phiXX);
const dot = (a: number[], c: Float64Array | number[]) => { let s = 0; for (let j = 0; j < M; j++) s += a[j] * c[j]; return s; };

/** 본문 pinn_system: 손실 = ‖A c − rhs‖² */
function system(n: number, lam: number) {
  const xc = Array.from({ length: n }, (_, i) => (n === 1 ? 0 : i / (n - 1)));
  const s = Math.sqrt(n), sb = Math.sqrt(lam / 2);
  const A = [...xc.map((x) => phiXX(x).map((v) => -v / s)), phi(0).map((v) => sb * v), phi(1).map((v) => sb * v)];
  const rhs = [...xc.map((x) => fSrc(x) / s), 0, 0];
  return { xc, A, rhs };
}

/** 한쪽 Jacobi 특잇값 분해로 최소 노름 최소제곱 해. 특잇값 < eps·max(m, n)·σ_max는 버린다(numpy lstsq의 rcond=None). */
function lstsq(A: number[][], rhs: number[]) {
  const m = A.length;
  const U = Array.from({ length: M }, (_, j) => Float64Array.from(A, (row) => row[j])); // 열 j, (m,)
  const V = Array.from({ length: M }, (_, j) => { const v = new Float64Array(M); v[j] = 1; return v; });
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < M - 1; p++) for (let q = p + 1; q < M; q++) {
      const up = U[p], uq = U[q];
      let a = 0, b = 0, g = 0;
      for (let i = 0; i < m; i++) { a += up[i] * up[i]; b += uq[i] * uq[i]; g += up[i] * uq[i]; }
      if (Math.abs(g) <= 1e-15 * Math.sqrt(a * b) || g === 0) continue;
      off = Math.max(off, Math.abs(g) / Math.sqrt(a * b));
      const zeta = (b - a) / (2 * g);
      const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), s = c * t;
      for (let i = 0; i < m; i++) { const x = up[i], y = uq[i]; up[i] = c * x - s * y; uq[i] = s * x + c * y; }
      const vp = V[p], vq = V[q];
      for (let i = 0; i < M; i++) { const x = vp[i], y = vq[i]; vp[i] = c * x - s * y; vq[i] = s * x + c * y; }
    }
    if (off < 1e-15) break;
  }
  const sig = U.map((u) => Math.sqrt(u.reduce((acc, v) => acc + v * v, 0)));
  const smax = Math.max(...sig), cut = 2.220446049250313e-16 * Math.max(m, M) * smax;
  const c = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    if (sig[j] <= cut) continue;
    let ub = 0; for (let i = 0; i < m; i++) ub += U[j][i] * rhs[i];
    const k = ub / (sig[j] * sig[j]);
    for (let i = 0; i < M; i++) c[i] += k * V[j][i];
  }
  return { c, smax };
}

/** 본문 gradient_descent: c ← c − lr·Aᵀ(A c − rhs), lr = 1/λ_max(AᵀA). AᵀA와 Aᵀrhs를 미리 만들어 같은 갱신을 한다. */
function gradientDescent(A: number[][], rhs: number[], smax: number, steps: number) {
  const H = new Float64Array(M * M), g = new Float64Array(M);
  for (const [r, row] of A.entries()) for (let i = 0; i < M; i++) { g[i] += row[i] * rhs[r]; for (let j = 0; j < M; j++) H[i * M + j] += row[i] * row[j]; }
  const lr = 1 / (smax * smax);
  const c = new Float64Array(M), grad = new Float64Array(M);
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < M; i++) { let acc = -g[i]; const o = i * M; for (let j = 0; j < M; j++) acc += H[o + j] * c[j]; grad[i] = acc; }
    for (let i = 0; i < M; i++) c[i] -= lr * grad[i];
  }
  return { c, lr };
}

export function solve(n: number, lam: number, solver: "ls" | "gd") {
  const { xc, A, rhs } = system(n, lam);
  const ls = lstsq(A, rhs);
  const { c, lr } = solver === "ls" ? { c: ls.c, lr: NaN } : gradientDescent(A, rhs, ls.smax, 20000);
  const res = (pxx: number[], x: number) => -dot(pxx, c) - fSrc(x);
  const rTrain = Math.sqrt(xc.reduce((s, x) => s + res(phiXX(x), x) ** 2, 0) / n);
  const resT = XT.map((x, i) => res(PXT[i], x));
  const rTest = Math.sqrt(resT.reduce((s, v) => s + v * v, 0) / NT);
  const u = PT.map((p) => dot(p, c));
  const bc = Math.max(Math.abs(u[0]), Math.abs(u[NT - 1]));
  const err = Math.max(...u.map((v, i) => Math.abs(v - uTrue(XT[i]))));
  const resC = xc.map((x) => res(phiXX(x), x));
  return { xc, u, resT, resC, rTrain, rTest, bc, err, lr };
}

const e1 = (v: number) => v.toExponential(1).replace(/e([+-])(\d)$/, (_, sg: string, d: string) => `e${sg}0${d}`); // 2.5e+01 꼴

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "PINN 손실 균형", "manipulate", narrow ? 1.0 : 0.6, "정확해와 PINN 해, 그리고 위치별 방정식 잔차의 크기(로그 축)");
  let solver: "ls" | "gd" = attrs.solver === "gd" ? "gd" : "ls";
  let n = 15, logLam = 0;
  let out = solve(n, 10 ** logLam, solver);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const axL = 46, R = 12;
    const X = (x: number) => axL + x * (W - axL - R);
    // 위: 해
    const t1 = 26, b1 = H * 0.5 - 16;
    const umin = Math.max(-2.5, Math.min(-0.4, ...out.u)), umax = Math.min(3.5, Math.max(1.4, ...out.u));
    const Y1 = (v: number) => b1 - ((Math.max(umin, Math.min(umax, v)) - umin) / (umax - umin)) * (b1 - t1);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText("해 u(x)", 6, 16);
    ctx.font = "11px Pretendard Variable, sans-serif";
    const leg: [string, string][] = [["정확해 u*", C.mid], ["PINN 해", C.edge]];
    let lx = narrow ? 64 : 80;
    for (const [s, col] of leg) { ctx.fillStyle = col; ctx.fillRect(lx, 9, 14, 3); ctx.fillStyle = C.mid; ctx.fillText(s, lx + 18, 15); lx += 28 + ctx.measureText(s).width; }
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const v of [0, 1]) {
      ctx.beginPath(); ctx.moveTo(axL, Y1(v)); ctx.lineTo(W - R, Y1(v)); ctx.stroke();
      ctx.fillText(String(v), axL - 6 - ctx.measureText(String(v)).width, Y1(v) + 4);
    }
    const curve = (ys: number[], Y: (v: number) => number) => { ctx.beginPath(); ys.forEach((v, i) => (i ? ctx.lineTo(X(XT[i]), Y(v)) : ctx.moveTo(X(XT[i]), Y(v)))); ctx.stroke(); };
    ctx.strokeStyle = C.mid; ctx.lineWidth = 4; ctx.globalAlpha = 0.5;
    curve(XT.map(uTrue), Y1);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    curve(out.u, Y1);
    // collocation 점 표시
    ctx.fillStyle = C.ink2;
    for (const x of out.xc) { ctx.beginPath(); ctx.moveTo(X(x), b1 + 2); ctx.lineTo(X(x) - 3.5, b1 + 9); ctx.lineTo(X(x) + 3.5, b1 + 9); ctx.fill(); }
    // 아래: |잔차| (로그 축)
    const t2 = H * 0.5 + 22, b2 = H - 22;
    const L0 = -8, L1 = 3;
    const Y2 = (v: number) => { const l = Math.log10(Math.max(Math.abs(v), 10 ** L0)); return b2 - ((Math.min(l, L1) - L0) / (L1 - L0)) * (b2 - t2); };
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText("|잔차 −u'' − f| (로그 축)", 6, t2 - 8);
    ctx.font = "11px JetBrains Mono, monospace";
    for (const p of [-8, -4, 0, 3]) {
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(axL, Y2(10 ** p)); ctx.lineTo(W - R, Y2(10 ** p)); ctx.stroke();
      ctx.fillStyle = C.mid; const s = `1e${p}`;
      ctx.fillText(s, axL - 6 - ctx.measureText(s).width, Y2(10 ** p) + 4);
    }
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.4;
    curve(out.resT, Y2);
    ctx.fillStyle = C.edge;
    out.xc.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y2(out.resC[i]), 3.2, 0, 7); ctx.fill(); });
    ctx.fillStyle = C.mid;
    for (const x of [0, 0.5, 1]) { const s = String(x); ctx.fillText(s, Math.min(X(x) - ctx.measureText(s).width / 2, W - R - ctx.measureText(s).width), b2 + 15); }
  }

  let pending = 0;
  function update() {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      out = solve(n, 10 ** logLam, solver);
      ro.set("train", `학습점 residual ${out.rTrain < 1e-8 ? "< 1e-8" : e1(out.rTrain)}`);
      ro.set("test", `새 점 residual ${e1(out.rTest)}`, out.rTest > 100 * Math.max(out.rTrain, 1e-6) ? "hot" : "");
      ro.set("bc", `경계 위반 ${e1(out.bc)}`, out.bc > 1e-2 ? "hot" : "");
      ro.set("err", `최대 해 오차 ${out.err < 1e-6 ? "< 1e-6" : e1(out.err)}`, out.err > 0.05 ? "hot" : out.err < 1e-3 ? "ok" : "");
      ro.set("lr", solver === "gd" ? `학습률 ${e1(out.lr)}` : "손실의 정확한 최소점");
      draw();
    });
  }

  const sBtns = presets<"ls" | "gd">([{ key: "ls", label: "최소제곱" }, { key: "gd", label: "경사하강 2만 step" }], (k) => { solver = k; update(); }, solver);
  sBtns.row.classList.add("segmented");
  const pn = param({ label: "N_r", min: 4, max: 100, step: 1, value: n, fmt: (v) => `${v}점`, aria: "collocation 점 수", onInput: (v) => { n = v; update(); } });
  const pl = param({ label: "λ_bc", min: 0, max: 4, step: 0.25, value: logLam, fmt: (v) => (v === 0 ? "1" : `10^${v}`), aria: "경계 손실 가중치의 상용로그", onInput: (v) => { logLam = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row, pl.row);
  const ro = readouts();
  f.panel.append(sBtns.row, grid, ro.row);
  f.onResize(draw);
  update();
}
