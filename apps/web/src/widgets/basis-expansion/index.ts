// 기저 전개와 해상도 (data/widgets.toml: basis-expansion).
// 목표 함수 u(t), t ∈ [0, 1]을 Fourier basis(x = 2πt의 1, cos kx, sin kx) 또는 Legendre 다항식(x = 2t − 1의 P₀, P₁, …)
// 앞의 M개 항으로 전개한다. 계수는 격자 N점의 값에서 구적으로 계산한다(Fourier: 주기 격자의 이산 합, Legendre: 중점 규칙).
// 격자 해상도 N을 바꾸면 격자 값 벡터는 길이와 값이 모두 바뀌지만, 계수는 연속 함수에 묶여 있어 거의 그대로다.
// 아래 그림은 계수 크기 |c_j|(로그 눈금)이고, 얇은 눈금은 N = 4096에서 계산한 기준 계수다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export type Basis = "fourier" | "legendre";
export const TARGETS = {
  smooth: { label: "e^{sin x}", short: "e^sin", u: (t: number) => Math.exp(Math.sin(2 * Math.PI * t)) },
  square: { label: "사각파", short: "사각파", u: (t: number) => (t % 0.5 === 0 ? 0 : t < 0.5 ? 1 : -1) },
  ramp: { label: "경사", short: "경사", u: (t: number) => 2 * t - 1 },
  peak: { label: "좁은 봉우리", short: "봉우리", u: (t: number) => Math.exp(-0.5 * ((t - 0.5) / 0.04) ** 2) },
} as const;
export type TargetKey = keyof typeof TARGETS;

/** j번째 basis 함수의 t에서의 값. Fourier: j = 0 → 1, 2k−1 → cos kx, 2k → sin kx. Legendre: P_j(2t − 1). */
export function basisValues(basis: Basis, M: number, t: number): number[] {
  const out = new Array<number>(M);
  if (basis === "fourier") {
    const x = 2 * Math.PI * t;
    out[0] = 1;
    for (let j = 1; j < M; j++) { const k = Math.ceil(j / 2); out[j] = j % 2 ? Math.cos(k * x) : Math.sin(k * x); }
  } else {
    const x = 2 * t - 1;
    out[0] = 1;
    if (M > 1) out[1] = x;
    for (let n = 1; n + 1 < M; n++) out[n + 1] = ((2 * n + 1) * x * out[n] - n * out[n - 1]) / (n + 1);
  }
  return out;
}

/** 격자 N점의 값으로 계수 M개를 구한다. Fourier는 주기 격자 t_i = i/N, Legendre는 중점 t_i = (i + 1/2)/N. */
export function coeffs(key: TargetKey, basis: Basis, M: number, N: number): number[] {
  const u = TARGETS[key].u;
  const c = new Array<number>(M).fill(0);
  for (let i = 0; i < N; i++) {
    const t = basis === "fourier" ? i / N : (i + 0.5) / N;
    const ui = u(t), phi = basisValues(basis, M, t);
    for (let j = 0; j < M; j++) c[j] += ui * phi[j];
  }
  for (let j = 0; j < M; j++) {
    // Fourier: ⟨u, 1⟩/⟨1, 1⟩ = 평균, ⟨u, cos⟩/⟨cos, cos⟩ = 2 × 평균. Legendre: (2j + 1)/2 ∫ u P_j dx, dx = 2/N
    c[j] *= basis === "fourier" ? (j === 0 ? 1 : 2) / N : (2 * j + 1) / N;
  }
  return c;
}

export function evalSeries(basis: Basis, c: number[], t: number) {
  const phi = basisValues(basis, c.length, t);
  let s = 0;
  for (let j = 0; j < c.length; j++) s += c[j] * phi[j];
  return s;
}

/** [0, 1]에서의 L² 오차(중점 2000점)와 최대 오차 */
export function errors(key: TargetKey, basis: Basis, c: number[]) {
  const u = TARGETS[key].u, Q = 2000;
  let l2 = 0, mx = 0;
  for (let i = 0; i < Q; i++) {
    const t = (i + 0.5) / Q, e = u(t) - evalSeries(basis, c, t);
    l2 += (e * e) / Q;
    mx = Math.max(mx, Math.abs(e));
  }
  return { l2: Math.sqrt(l2), max: mx };
}

/** 기저 함수 중 가장 높은 Fourier 파수(aliasing 판정용) */
export const maxWave = (M: number) => Math.floor(M / 2);

const NS = [8, 16, 32, 64, 128, 256, 512];
const sci = (v: number) => (v === 0 ? "0" : v.toExponential(1).replace("e-", "e−"));

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 480;
  const f = frame(root, "기저 전개와 해상도", "manipulate", narrow ? 1.0 : 0.62, "목표 함수와 M항 기저 근사, 격자 표본점, 그리고 계수 크기의 막대그래프");
  let key: TargetKey = attrs.target in TARGETS ? (attrs.target as TargetKey) : "smooth";
  let basis: Basis = attrs.basis === "legendre" ? "legendre" : "fourier";
  let M = 9, nIdx = 2;
  let c: number[] = [], ref: number[] = [];
  let curve: number[] = [], approx: number[] = [];

  function compute() {
    const N = NS[nIdx];
    c = coeffs(key, basis, M, N);
    ref = coeffs(key, basis, M, 4096);
    const P = 400;
    curve = []; approx = [];
    for (let i = 0; i <= P; i++) { const t = i / P; curve.push(TARGETS[key].u(t)); approx.push(evalSeries(basis, c, t)); }
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const padL = 40, padR = 14, topH = Math.round(H * 0.6), padT = 28;
    // 위: 함수
    let lo = Math.min(...curve), hi = Math.max(...curve);
    const span = hi - lo || 1;
    lo -= 0.18 * span; hi += 0.18 * span;
    const X = (t: number) => padL + t * (W - padL - padR);
    const Y = (v: number) => padT + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (topH - padT - 22);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, W - padL - padR, topH - padT - 22);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    if (lo < 0 && hi > 0) {
      ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
      ctx.fillText("0", padL - 6 - ctx.measureText("0").width, Y(0) + 4);
    }
    ctx.fillText("t = 0", padL, topH - 8);
    ctx.fillText("1", W - padR - ctx.measureText("1").width, topH - 8);
    const line = (vals: number[], color: string, w: number) => {
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineJoin = "round"; ctx.beginPath();
      vals.forEach((v, i) => { const x = X(i / (vals.length - 1)), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    };
    line(curve, C.mid, 2);
    line(approx, C.edge, 2.2);
    // 격자 표본점
    const N = NS[nIdx];
    if (N <= 128) {
      ctx.fillStyle = C.ink2;
      for (let i = 0; i < N; i++) {
        const t = basis === "fourier" ? i / N : (i + 0.5) / N;
        ctx.beginPath(); ctx.arc(X(t), Y(TARGETS[key].u(t)), N > 64 ? 1.6 : 2.6, 0, 7); ctx.fill();
      }
    }
    // 범례
    ctx.font = "600 12px Pretendard, sans-serif";
    const leg: [string, string][] = [[C.mid, "u(t)"], [C.edge, `M = ${M}항 근사`], [C.ink2, `격자 ${N}점`]];
    let lx = padL;
    for (const [col, txt] of leg) {
      ctx.fillStyle = col; ctx.fillRect(lx, 11, 12, 3);
      ctx.fillStyle = C.ink; ctx.fillText(txt, lx + 16, 16);
      lx += 16 + ctx.measureText(txt).width + 14;
    }

    // 아래: 계수 크기 (로그 눈금 10^-16 ~ 10^1)
    const bT = topH + 4, bB = H - 22, bH = bB - bT;
    const LMIN = -16, LMAX = 1;
    const BY = (a: number) => bT + (1 - (Math.max(LMIN, Math.min(LMAX, Math.log10(Math.max(a, 1e-300)))) - LMIN) / (LMAX - LMIN)) * bH;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(padL, bT, W - padL - padR, bH);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const e of [0, -8, -16]) {
      const y = BY(10 ** e);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      const s = e === 0 ? "1" : `1e${e}`.replace("-", "−");
      ctx.fillText(s, padL - 4 - ctx.measureText(s).width, y + 4);
    }
    const slot = (W - padL - padR) / M, bw = Math.max(2, Math.min(18, slot * 0.6));
    for (let j = 0; j < M; j++) {
      const cx = padL + (j + 0.5) * slot;
      const y = BY(Math.abs(c[j]));
      ctx.fillStyle = alpha(C.edge, 0.85);
      ctx.fillRect(cx - bw / 2, y, bw, bB - y);
      const yr = BY(Math.abs(ref[j]));
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 2, yr); ctx.lineTo(cx + bw / 2 + 2, yr); ctx.stroke();
    }
    ctx.fillStyle = C.mid;
    ctx.fillText(`|c_j|, j = 0…${M - 1}`, padL, H - 6);
  }

  function update() {
    compute();
    const N = NS[nIdx];
    const e = errors(key, basis, c);
    let dc = 0;
    for (let j = 0; j < M; j++) dc = Math.max(dc, Math.abs(c[j] - ref[j]));
    ro.set("dim", `격자 값 ${N}개 → 계수 ${M}개`);
    ro.set("l2", `L² 오차 ${sci(e.l2)}`, "hot");
    ro.set("max", `최대 오차 ${sci(e.max)}`);
    ro.set("dc", `계수 차이(N = ${N} 대 4096) ${sci(dc)}`, dc < 1e-6 ? "ok" : "");
    const alias = basis === "fourier" ? 2 * maxWave(M) >= N : M > N;
    ro.set("alias", alias ? (basis === "fourier" ? "파수 ≥ N/2: aliasing" : "항 수 > 격자 점 수") : "격자가 항을 표현함", alias ? "hot" : "ok");
    draw();
  }

  const tg = presets((Object.keys(TARGETS) as TargetKey[]).map((k) => ({ key: k, label: narrow ? TARGETS[k].short : TARGETS[k].label })), (k) => { key = k; update(); }, key);
  const bs = presets<Basis>([{ key: "fourier", label: "Fourier" }, { key: "legendre", label: "Legendre" }], (k) => { basis = k; update(); }, basis);
  const pm = param({ label: "M", min: 1, max: 41, step: 1, value: M, fmt: (v) => `${v}`, aria: "basis 항 수 M", onInput: (v) => { M = v; update(); } });
  const pn = param({ label: "N", min: 0, max: NS.length - 1, step: 1, value: nIdx, fmt: (v) => `${NS[v]}`, aria: "격자 점 수 N", onInput: (v) => { nIdx = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pm.row, pn.row);
  const ro = readouts();
  f.panel.append(tg.row, bs.row, grid, ro.row);
  f.onResize(draw);
  update();
}
