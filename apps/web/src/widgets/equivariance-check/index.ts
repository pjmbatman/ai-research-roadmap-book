// 등변성 검사 (data/widgets.toml: equivariance-check).
// 7.10장 실습처럼 점 6개짜리 2차원 점구름(학습 데이터는 x 방향으로 길쭉: 표준편차 x 2.0, y 0.5)에서
// 벡터 목표 v(X) = (1/N) Σ ‖x_n − c‖² (x_n − c)를 세 모델로 학습하고, 회전 R에 대해 f(RX)와 R f(X)를 비교한다.
//  - 일반 MLP: 점마다 같은 MLP(2 → tanh 은닉 64개 → 2, 은닉층은 고정 무작위·출력층은 ridge 최소제곱)를 적용해 평균
//  - 증강 학습: 같은 구조를, 무작위로 돌린 입력과 함께 돌린 정답으로 학습
//  - 등변 구성: v̂ = (1/N) Σ g(r_n)(x_n − c), g(r) = a₁r² + a₂r⁴ (최소제곱으로 학습)
// 점구름은 무게중심을 원점에 둔다(회전은 무게중심 둘레). 계산은 모두 브라우저에서 실제로 한다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";

const NP = 6, HID = 64, NTRAIN = 400, NAUG = 4;
type Cloud = Float64Array; // [x0, y0, x1, y1, ...] 길이 12, 무게중심 0
type Vec = [number, number];

function makeCloud(r: ReturnType<typeof rng>): Cloud {
  const X = new Float64Array(2 * NP);
  let cx = 0, cy = 0;
  for (let n = 0; n < NP; n++) { X[2 * n] = 2.0 * r.normal(); X[2 * n + 1] = 0.5 * r.normal(); cx += X[2 * n] / NP; cy += X[2 * n + 1] / NP; }
  for (let n = 0; n < NP; n++) { X[2 * n] -= cx; X[2 * n + 1] -= cy; }
  return X;
}
export function rotate(X: Cloud, th: number): Cloud {
  const c = Math.cos(th), s = Math.sin(th), Y = new Float64Array(X.length);
  for (let n = 0; n < X.length / 2; n++) { Y[2 * n] = c * X[2 * n] - s * X[2 * n + 1]; Y[2 * n + 1] = s * X[2 * n] + c * X[2 * n + 1]; }
  return Y;
}
const rotV = (v: Vec, th: number): Vec => [Math.cos(th) * v[0] - Math.sin(th) * v[1], Math.sin(th) * v[0] + Math.cos(th) * v[1]];
/** 정답: v(X) = (1/N) Σ ‖x_n‖² x_n (무게중심 0) */
export function target(X: Cloud): Vec {
  let a = 0, b = 0;
  for (let n = 0; n < NP; n++) { const x = X[2 * n], y = X[2 * n + 1], r2 = x * x + y * y; a += (r2 * x) / NP; b += (r2 * y) / NP; }
  return [a, b];
}

/** 대칭 양의 정부호 A (n×n, 행 우선)로 A z = B (n×m)를 Cholesky로 푼다. */
export function cholSolve(A: Float64Array, B: Float64Array, n: number, m: number) {
  const Lm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= Lm[i * n + k] * Lm[j * n + k];
    Lm[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / Lm[j * n + j];
  }
  const Z = new Float64Array(B);
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) { let s = Z[i * m + c]; for (let k = 0; k < i; k++) s -= Lm[i * n + k] * Z[k * m + c]; Z[i * m + c] = s / Lm[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = Z[i * m + c]; for (let k = i + 1; k < n; k++) s -= Lm[k * n + i] * Z[k * m + c]; Z[i * m + c] = s / Lm[i * n + i]; }
  }
  return Z;
}

export type ModelKey = "mlp" | "aug" | "equi";
export interface Models { predict: Record<ModelKey, (X: Cloud) => Vec>; g: [number, number] }

/** 세 모델을 학습한다(시드 고정). 정답에는 표준편차 0.2의 관측 잡음을 더해 학습한다. */
export function train(seed = 7): Models {
  const r = rng(seed);
  // 일반 MLP: 점마다 같은 MLP h(x_n)(입력 2 → tanh 은닉 HID → 출력 2)을 적용해 평균한다.
  // 은닉층 가중치는 고정 무작위, 출력층은 ridge 최소제곱으로 학습한다. 점 순서에는 무관하지만 회전에 대한 구조는 없다.
  const F = HID + 3;
  const W = Float64Array.from({ length: HID * 2 }, () => 0.6 * r.normal());
  const bh = Float64Array.from({ length: HID }, () => 1.5 * r.normal());
  const feat = (X: Cloud, out: Float64Array, o: number) => {
    out.fill(0, o, o + F);
    for (let n = 0; n < NP; n++) {
      const x = X[2 * n], y = X[2 * n + 1];
      for (let h = 0; h < HID; h++) out[o + h] += Math.tanh(W[2 * h] * x + W[2 * h + 1] * y + bh[h]) / NP;
      out[o + HID] += x / NP; out[o + HID + 1] += y / NP;
    }
    out[o + HID + 2] = 1;
  };
  const clouds = Array.from({ length: NTRAIN }, () => makeCloud(r));
  const noisy = (v: Vec): Vec => [v[0] + 0.2 * r.normal(), v[1] + 0.2 * r.normal()];
  const fit = (rows: { X: Cloud; y: Vec }[]) => {
    const n = rows.length, Phi = new Float64Array(n * F);
    rows.forEach((row, i) => feat(row.X, Phi, i * F));
    const A = new Float64Array(F * F), B = new Float64Array(F * 2);
    for (let i = 0; i < n; i++) {
      const o = i * F, y = rows[i].y;
      for (let a = 0; a < F; a++) {
        const pa = Phi[o + a];
        for (let b = 0; b <= a; b++) A[a * F + b] += pa * Phi[o + b];
        B[a * 2] += pa * y[0]; B[a * 2 + 1] += pa * y[1];
      }
    }
    for (let a = 0; a < F; a++) { for (let b = 0; b < a; b++) A[b * F + a] = A[a * F + b]; A[a * F + a] += 1e-6 * n; }
    const beta = cholSolve(A, B, F, 2);
    const tmp = new Float64Array(F);
    return (X: Cloud): Vec => {
      feat(X, tmp, 0);
      let p = 0, q = 0;
      for (let a = 0; a < F; a++) { p += tmp[a] * beta[a * 2]; q += tmp[a] * beta[a * 2 + 1]; }
      return [p, q];
    };
  };
  const mlp = fit(clouds.map((X) => ({ X, y: noisy(target(X)) })));
  const augRows: { X: Cloud; y: Vec }[] = [];
  for (const X of clouds) for (let k = 0; k < NAUG; k++) { const th = 2 * Math.PI * r.u(); const RX = rotate(X, th); augRows.push({ X: RX, y: noisy(target(RX)) }); }
  const aug = fit(augRows);
  // 등변 구성: v̂ = (1/N) Σ (a₁ r_n² + a₂ r_n⁴) x_n. 불변 가중치 × 등변 벡터라 어떤 a에서도 등변이다. 2×2 정규방정식.
  const moments = (X: Cloud): Vec[] => [1, 2].map((p) => {
    let a = 0, b = 0;
    for (let n = 0; n < NP; n++) { const x = X[2 * n], y = X[2 * n + 1], w = (x * x + y * y) ** p / NP; a += w * x; b += w * y; }
    return [a, b] as Vec;
  });
  const G = new Float64Array(4), h = new Float64Array(2);
  for (const X of clouds) {
    const m = moments(X), y = noisy(target(X));
    for (let i = 0; i < 2; i++) { h[i] += m[i][0] * y[0] + m[i][1] * y[1]; for (let j = 0; j < 2; j++) G[i * 2 + j] += m[i][0] * m[j][0] + m[i][1] * m[j][1]; }
  }
  const a = cholSolve(G, h, 2, 1);
  const equi = (X: Cloud): Vec => { const m = moments(X); return [a[0] * m[0][0] + a[1] * m[1][0], a[0] * m[0][1] + a[1] * m[1][1]]; };
  return { predict: { mlp, aug, equi }, g: [a[0], a[1]] };
}

/** 등변성 오차 ‖f(RX) − R f(X)‖ */
export function eqErr(f: (X: Cloud) => Vec, X: Cloud, th: number) {
  const a = f(rotate(X, th)), b = rotV(f(X), th);
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
export function testCloud(seed: number) { return makeCloud(rng(1000 + seed)); }

const MODELS: { key: ModelKey; label: string }[] = [
  { key: "mlp", label: "일반 MLP" },
  { key: "aug", label: "증강 학습" },
  { key: "equi", label: "등변 구성" },
];
const NA = 181;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 520;
  const f = frame(root, "등변성 검사", "compare", narrow ? 1.35 : 0.5, "회전한 점구름과 f(Rx), R f(x) 두 화살표, 그리고 회전각에 따른 세 모델의 등변성 오차 곡선");
  const models = train();
  let model: ModelKey = (MODELS.find((m) => m.key === attrs.model)?.key ?? "mlp");
  let deg = 60, seed = 1;
  let X = testCloud(seed);
  let curves: Record<ModelKey, number[]> = { mlp: [], aug: [], equi: [] };
  const col = (k: ModelKey) => (k === "mlp" ? C.edge : k === "aug" ? C.ink2 : C.good);

  function sweep() {
    for (const m of MODELS) curves[m.key] = Array.from({ length: NA }, (_, i) => eqErr(models.predict[m.key], X, (2 * Math.PI * i) / (NA - 1)));
  }

  function arrow(a: Vec, b: Vec, color: string, w: number) {
    const { ctx } = f;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.beginPath(); ctx.moveTo(...b);
    ctx.lineTo(b[0] - 11 * Math.cos(ang - 0.4), b[1] - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(b[0] - 11 * Math.cos(ang + 0.4), b[1] - 11 * Math.sin(ang + 0.4));
    ctx.fill();
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const th = (deg * Math.PI) / 180;
    // 왼쪽(좁으면 위): 점구름과 두 화살표
    const side = narrow ? Math.min(W - 16, H * 0.55) : Math.min(H - 16, W * 0.46);
    const ox = narrow ? W / 2 : 8 + side / 2, oy = narrow ? 8 + side / 2 : H / 2;
    const rad = side / 2 - 8;
    let rmax = 0;
    for (let n = 0; n < NP; n++) rmax = Math.max(rmax, Math.hypot(X[2 * n], X[2 * n + 1]));
    const sp = (0.62 * rad) / rmax;
    const v0 = target(X), vs = (0.9 * rad) / Math.max(Math.hypot(...v0), 1e-9);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(ox, oy, rad, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox - rad, oy); ctx.lineTo(ox + rad, oy); ctx.moveTo(ox, oy - rad); ctx.lineTo(ox, oy + rad); ctx.stroke();
    // 회전각 호
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(ox, oy, 18, 0, -th, true); ctx.stroke();
    const RX = rotate(X, th);
    ctx.fillStyle = alpha(C.mid, 0.35);
    for (let n = 0; n < NP; n++) { ctx.beginPath(); ctx.arc(ox + sp * X[2 * n], oy - sp * X[2 * n + 1], 3.5, 0, 7); ctx.fill(); }
    ctx.fillStyle = C.ink;
    for (let n = 0; n < NP; n++) { ctx.beginPath(); ctx.arc(ox + sp * RX[2 * n], oy - sp * RX[2 * n + 1], 4.5, 0, 7); ctx.fill(); }
    const P = models.predict[model];
    const a = P(RX), b = rotV(P(X), th);
    const tip = (v: Vec): Vec => [ox + vs * v[0], oy - vs * v[1]];
    arrow([ox, oy], tip(b), C.ink2, 3);
    arrow([ox, oy], tip(a), C.edge, 2.2);
    ctx.setLineDash([4, 3]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(...tip(a)); ctx.lineTo(...tip(b)); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = "600 12px JetBrains Mono, monospace";
    [[C.ink2, "R f(x)"], [C.edge, "f(Rx)"]].forEach(([color, s], i) => {
      const lx = Math.max(6, ox - rad - 4), ly = Math.max(14, oy - rad + 6) + 16 * i;
      ctx.fillStyle = color; ctx.fillRect(lx, ly - 5, 14, 3);
      ctx.fillText(s, lx + 19, ly);
    });

    // 오른쪽(좁으면 아래): 회전각에 따른 오차 곡선
    const px0 = narrow ? 44 : ox + side / 2 + 52, px1 = W - 12;
    const py0 = narrow ? side + 34 : 30, py1 = H - 26;
    let ymax = 0;
    for (const m of MODELS) for (const v of curves[m.key]) ymax = Math.max(ymax, v);
    ymax = ymax || 1;
    const top = py0 + 26; // 범례 줄
    const XA = (d: number) => px0 + (d / 360) * (px1 - px0), YA = (v: number) => py1 - (v / ymax) * (py1 - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(px0, py0, px1 - px0, py1 - py0);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const d of [0, 90, 180, 270, 360]) { const s = `${d}°`; ctx.fillText(s, Math.min(XA(d) - ctx.measureText(s).width / 2, px1 - ctx.measureText(s).width), py1 + 14); }
    ctx.fillText("0", px0 - 10, py1 + 4);
    const ys = ymax >= 10 ? ymax.toFixed(0) : ymax.toFixed(1);
    ctx.fillText(ys, px0 - 6 - ctx.measureText(ys).width, YA(ymax) + 4);
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(px0, YA(ymax)); ctx.lineTo(px1, YA(ymax)); ctx.stroke();
    ctx.font = "600 12px Pretendard, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText("‖f(Rx) − R f(x)‖", px0, py0 - 8);
    for (const m of MODELS) {
      ctx.strokeStyle = col(m.key); ctx.lineWidth = m.key === model ? 2.6 : 1.3;
      ctx.globalAlpha = m.key === model ? 1 : 0.55;
      ctx.beginPath();
      curves[m.key].forEach((v, i) => { const x = XA((360 * i) / (NA - 1)), y = YA(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = C.mid; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(XA(deg), top); ctx.lineTo(XA(deg), py1); ctx.stroke(); ctx.setLineDash([]);
    for (const m of MODELS) {
      const e = eqErr(models.predict[m.key], X, th);
      ctx.fillStyle = col(m.key); ctx.beginPath(); ctx.arc(XA(deg), YA(e), m.key === model ? 5 : 3.5, 0, 7); ctx.fill();
    }
    // 범례
    ctx.font = "600 12px Pretendard, sans-serif";
    let lx = px0 + 8;
    for (const m of MODELS) {
      ctx.fillStyle = col(m.key); ctx.fillRect(lx, py0 + 12, 12, 3);
      ctx.fillStyle = C.ink; ctx.fillText(m.label, lx + 16, py0 + 17);
      lx += 16 + ctx.measureText(m.label).width + 12;
    }
  }

  function update() {
    const th = (deg * Math.PI) / 180;
    const P = models.predict[model];
    const e = eqErr(P, X, th), nb = Math.hypot(...rotV(P(X), th));
    const mean = curves[model].slice(0, NA - 1).reduce((s, v) => s + v, 0) / (NA - 1);
    const fmt = (v: number) => (v < 1e-9 ? "< 1e−9" : v < 0.01 ? v.toExponential(1).replace("e-", "e−") : v.toFixed(2));
    ro.set("e", `‖f(Rx) − R f(x)‖ = ${fmt(e)}`, "hot");
    ro.set("rel", `상대 ${fmt(e / Math.max(nb, 1e-12))}`);
    ro.set("mean", `360° 평균 ${fmt(mean)}`);
    const t0 = target(X), p0 = P(X);
    ro.set("fit", `θ = 0 예측 오차 ${fmt(Math.hypot(p0[0] - t0[0], p0[1] - t0[1]))}`);
    ro.set("kind", model === "equi" ? "구조로 보장" : model === "aug" ? "손실로 권장" : "보장 없음", model === "equi" ? "ok" : "");
    draw();
  }

  const pm = presets(MODELS, (k) => { model = k; update(); }, model);
  const pr = param({ label: "θ", min: 0, max: 360, step: 1, value: deg, fmt: (v) => `${v}°`, aria: "회전각 θ(도)", onInput: (v) => { deg = v; update(); } });
  pr.row.style.gridTemplateColumns = "1.4em minmax(0, 1fr) 3.4em";
  const grid = el("div", "param-grid");
  grid.append(pr.row);
  const tools = el("div", "wframe-row");
  const again = el("button", "preset");
  again.type = "button";
  again.textContent = "↻ 새 test 점구름";
  again.addEventListener("click", () => { seed++; X = testCloud(seed); sweep(); update(); });
  tools.append(again);
  const ro = readouts();
  f.panel.append(pm.row, grid, tools, ro.row);
  sweep();
  f.onResize(draw);
  update();
}
