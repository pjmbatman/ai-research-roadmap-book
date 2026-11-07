// forward vs reverse KL (data/widgets.toml: kl-direction). 14.1장의 예제를 그대로 쓴다.
// 참 분포 p = ½N(−d, 0.6²) + ½N(+d, 0.6²)에 가우시안 q = N(μ, σ²) 하나를 맞춘다.
// forward KL D(p‖q)는 기댓값을 p로 재므로 두 봉우리를 덮고(모멘트 맞추기), reverse KL D(q‖p)는 q로 재므로 한 봉우리에 숨는다.
// 위: p와 고른 방향의 최적 q(다른 방향의 최적 q는 점선). 아래: 각 μ에서 σ를 최적화한 KL, min_σ D(μ). reverse는 d가 커지면 우물이 둘로 갈라진다.
// KL은 [−10, 10]의 격자에서 수치적분한다(장의 코드와 같은 방식). log p는 로그 공간에서 계산해 꼬리의 underflow를 피한다.
import { frame, param, presets, readouts, el, C } from "../frame";

export type Dir = "fwd" | "rev";
const S0 = 0.6;
const G = 801, XA = -10, XB = 10, DX = (XB - XA) / (G - 1);
const XS = Float64Array.from({ length: G }, (_, i) => XA + i * DX);
const LOG2PI = Math.log(2 * Math.PI);

export function target(d: number) {
  const logp = new Float64Array(G), p = new Float64Array(G);
  for (let i = 0; i < G; i++) {
    const a = -0.5 * ((XS[i] + d) / S0) ** 2, b = -0.5 * ((XS[i] - d) / S0) ** 2, m = Math.max(a, b);
    logp[i] = Math.log(0.5) - 0.5 * LOG2PI - Math.log(S0) + m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    p[i] = Math.exp(logp[i]);
  }
  // forward KL에 쓰는 p의 적분값(같은 격자의 합): Σp, Σp·x, Σp·x², Σp·log p
  let m0 = 0, m1 = 0, m2 = 0, plogp = 0;
  for (let i = 0; i < G; i++) { m0 += p[i]; m1 += p[i] * XS[i]; m2 += p[i] * XS[i] ** 2; if (p[i] > 0) plogp += p[i] * logp[i]; }
  return { logp, p, m0: m0 * DX, m1: m1 * DX, m2: m2 * DX, plogp: plogp * DX };
}
type Target = ReturnType<typeof target>;

/** D(p‖q) 또는 D(q‖p), q = N(mu, s²) [nat]. 둘 다 같은 격자의 수치적분이다. */
export function kl(t: Target, dir: Dir, mu: number, s: number): number {
  const c = -0.5 * LOG2PI - Math.log(s);
  if (dir === "fwd") {
    // Σ p (log p − log q) dx를 전개한 것: log q가 x의 이차식이라 p의 적분값으로 정확히 같은 합이 된다.
    return t.plogp - (c * t.m0 - (t.m2 - 2 * mu * t.m1 + mu * mu * t.m0) / (2 * s * s));
  }
  // q가 무시할 만한(10σ 밖) 격자점은 건너뛴다
  const i0 = Math.max(0, Math.floor((mu - 10 * s - XA) / DX)), i1 = Math.min(G - 1, Math.ceil((mu + 10 * s - XA) / DX));
  let sum = 0;
  for (let i = i0; i <= i1; i++) {
    const lq = c - 0.5 * ((XS[i] - mu) / s) ** 2;
    sum += Math.exp(lq) * (lq - t.logp[i]);
  }
  return sum * DX;
}

const golden = (fn: (v: number) => number, a: number, b: number, it = 40) => {
  const r = (Math.sqrt(5) - 1) / 2;
  let c = b - r * (b - a), d = a + r * (b - a), fc = fn(c), fd = fn(d);
  for (let i = 0; i < it; i++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - r * (b - a); fc = fn(c); }
    else { a = c; c = d; fc = fd; d = a + r * (b - a); fd = fn(d); }
  }
  const v = (a + b) / 2;
  return { arg: v, val: fn(v) };
};
/** 주어진 μ에서 σ를 최적화한 KL. log σ에 대해 황금분할 탐색한다. */
export const bestSigma = (t: Target, dir: Dir, mu: number, it = 32) => {
  const g = golden((ls) => kl(t, dir, mu, Math.exp(ls)), Math.log(0.1), Math.log(6), it);
  return { s: Math.exp(g.arg), val: g.val };
};
export const MUS = Array.from({ length: 57 }, (_, i) => -3.5 + i * 0.125);

/** 최적 q: μ 격자에서 min_σ D(μ) 곡선을 구하고, 가장 낮은 칸 둘레에서 μ를 다시 다듬는다. 동점이면 왼쪽(μ ≤ 0)을 고른다. */
export function fit(t: Target, dir: Dir) {
  const curve = MUS.map((mu) => bestSigma(t, dir, mu, 24).val);
  let bi = 0;
  curve.forEach((v, i) => { if (v < curve[bi] - 1e-9) bi = i; });
  const lo = MUS[Math.max(0, bi - 1)], hi = MUS[Math.min(MUS.length - 1, bi + 1)];
  const g = golden((mu) => bestSigma(t, dir, mu).val, lo, hi, 30);
  const s = bestSigma(t, dir, g.arg).s;
  return { mu: g.arg, s, val: g.val, curve };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "forward vs reverse KL", "manipulate", narrow ? 0.9 : 0.62, "쌍봉 참 분포 p와 KL 방향별로 맞춘 가우시안 q, 그리고 μ에 따른 최소 KL 곡선");
  let d = 2.0, dir: Dir = "rev";
  let t = target(d);
  let res: Record<Dir, ReturnType<typeof fit>> = { fwd: fit(t, "fwd"), rev: fit(t, "rev") };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pad = 18, top = 14, split = Math.round(H * 0.62), base = split - 20;
    const V0 = -6, V1 = 6;
    const X = (v: number) => pad + ((v - V0) / (V1 - V0)) * (W - 2 * pad);
    const gauss = (v: number, mu: number, s: number) => Math.exp(-0.5 * ((v - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const pdf = (v: number) => 0.5 * gauss(v, -d, S0) + 0.5 * gauss(v, d, S0);
    const vs = Array.from({ length: 361 }, (_, i) => V0 + ((V1 - V0) * i) / 360);
    const cur = res[dir], other = res[dir === "fwd" ? "rev" : "fwd"];
    const ymax = Math.max(...vs.map(pdf), gauss(cur.mu, cur.mu, cur.s), gauss(other.mu, other.mu, other.s)) * 1.1;
    const Y = (v: number) => base - (v / ymax) * (base - top);
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, base); ctx.lineTo(W - pad, base); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let k = -6; k <= 6; k += 2) { const s = String(k); ctx.fillText(s, X(k) - ctx.measureText(s).width / 2, base + 13); }
    // p (면)
    ctx.fillStyle = C.fill;
    ctx.beginPath(); ctx.moveTo(X(V0), base);
    vs.forEach((v) => ctx.lineTo(X(v), Y(pdf(v))));
    ctx.lineTo(X(V1), base); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.6;
    ctx.beginPath(); vs.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(pdf(v))) : ctx.moveTo(X(v), Y(pdf(v))))); ctx.stroke();
    // 다른 방향의 최적 q (점선)
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    ctx.beginPath(); vs.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(gauss(v, other.mu, other.s))) : ctx.moveTo(X(v), Y(gauss(v, other.mu, other.s))))); ctx.stroke();
    ctx.setLineDash([]);
    // 고른 방향의 최적 q
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6;
    ctx.beginPath(); vs.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(gauss(v, cur.mu, cur.s))) : ctx.moveTo(X(v), Y(gauss(v, cur.mu, cur.s))))); ctx.stroke();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.ink2; ctx.fillText("p", pad + 4, top + 8);
    ctx.fillStyle = C.edge; ctx.fillText(dir === "fwd" ? "q* = argmin D(p‖q)" : "q* = argmin D(q‖p)", pad + 20, top + 8);
    // 아래: min_σ D(μ)
    const b0 = split + 16, b1 = H - 22;
    const M0 = MUS[0], M1 = MUS[MUS.length - 1];
    const XM = (v: number) => X(v);
    const cmax = Math.min(Math.max(...cur.curve), Math.min(...cur.curve) + 2.5) * 1.05;
    const YM = (v: number) => b1 - (Math.min(v, cmax) / cmax) * (b1 - b0 - 10);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(XM(M0), b1); ctx.lineTo(XM(M1), b1); ctx.stroke();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); cur.curve.forEach((v, i) => (i ? ctx.lineTo(XM(MUS[i]), YM(v)) : ctx.moveTo(XM(MUS[i]), YM(v)))); ctx.stroke();
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(XM(cur.mu), YM(cur.val), 5, 0, 7); ctx.fill();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("min_σ KL", XM(M0), b0 + 2);
    ctx.fillText("μ", XM(M1) + 6, b1 + 4);
    for (const k of [-3, 0, 3]) { const s = String(k); ctx.fillText(s, XM(k) - ctx.measureText(s).width / 2, b1 + 13); }
  }

  function update(recompute = true) {
    if (recompute) { t = target(d); res = { fwd: fit(t, "fwd"), rev: fit(t, "rev") }; }
    const cur = res[dir];
    const fwdVal = kl(t, "fwd", cur.mu, cur.s), revVal = kl(t, "rev", cur.mu, cur.s);
    const sdp = Math.sqrt(d * d + S0 * S0);
    ro.set("q", `q* = N(${cur.mu > -0.005 ? "+" : "−"}${Math.abs(cur.mu).toFixed(2)}, ${cur.s.toFixed(2)}²)`, "hot");
    ro.set("f", `D(p‖q*) ${fwdVal.toFixed(3)} nat`, dir === "fwd" ? "ok" : "");
    ro.set("r", `D(q*‖p) ${revVal.toFixed(3)} nat`, dir === "rev" ? "ok" : "");
    ro.set("p", `p의 표준편차 ${sdp.toFixed(2)}`);
    draw();
  }

  const dirBtns = presets<Dir>([{ key: "fwd", label: "forward D(p‖q)" }, { key: "rev", label: "reverse D(q‖p)" }], (k) => { dir = k; update(false); }, dir);
  dirBtns.row.classList.add("segmented");
  let raf = 0;
  const pd = param({ label: "d", min: 0, max: 3, step: 0.1, value: d, fmt: (v) => v.toFixed(1), aria: "봉우리 위치 ±d", onInput: (v) => { d = v; cancelAnimationFrame(raf); raf = requestAnimationFrame(() => update()); } });
  const grid = el("div", "param-grid");
  grid.append(pd.row);
  const ro = readouts();
  f.panel.append(dirBtns.row, grid, ro.row);
  f.onResize(draw);
  update(false);
}
