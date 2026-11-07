// 26.3장 셀 1–3과 같은 계산: 256칸 점성 Burgers(에너지 보존형 중심 유속 + Heun), 8칸 블록 필터로 만든 32칸 ū와 τ,
// 세 closure(Smagorinsky·기울기 모델·5특징 선형 회귀)의 a priori 평가와 거친 격자에 결합한 a posteriori 평가.
import { CLOSURE_IC } from "./data";

export const L = 2 * Math.PI, NF = 256, NC = 32, K = NF / NC, D = L / NC;
export type Closure = ((ub: Float64Array) => Float64Array) | null;

/** 시드 s의 초기 장: Σ_k a_k/k · sin(k x + φ_k), k = 1..5 */
export function ic(seed: number, N: number) {
  const c = CLOSURE_IC[seed];
  return Float64Array.from({ length: N }, (_, j) => {
    const x = (j * L) / N;
    let v = 0;
    for (let k = 1; k <= 5; k++) v += (c[2 * k - 2] / k) * Math.sin(k * x + c[2 * k - 1]);
    return v;
  });
}

function rhs(u: Float64Array, dx: number, nu: number, tau: Float64Array | null, out: Float64Array, F: Float64Array) {
  const N = u.length;
  for (let j = 0; j < N; j++) {
    const a = u[j], b = u[(j + 1) % N];
    F[j] = (a * a + a * b + b * b) / 6 - (nu * (b - a)) / dx;
    if (tau) F[j] += 0.5 * (tau[j] + tau[(j + 1) % N]);
  }
  for (let j = 0; j < N; j++) out[j] = -(F[j] - F[(j - 1 + N) % N]) / dx;
}

/** Heun 적분, 저장한 상태 nsave개 (각 (N,)) */
export function run(u0: Float64Array, dx: number, nu: number, model: Closure, T = 2.0, nsave = 20) {
  const N = u0.length;
  const n = Math.ceil(T / (0.3 * Math.min(dx / 3, (dx * dx) / (2 * nu + 1e-12))));
  const dt = T / n, every = Math.floor(n / nsave);
  const u = Float64Array.from(u0), u1 = new Float64Array(N), r1 = new Float64Array(N), r2 = new Float64Array(N), F = new Float64Array(N);
  const out: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    rhs(u, dx, nu, model ? model(u) : null, r1, F);
    for (let j = 0; j < N; j++) u1[j] = u[j] + dt * r1[j];
    rhs(u1, dx, nu, model ? model(u1) : null, r2, F);
    for (let j = 0; j < N; j++) u[j] = 0.5 * (u[j] + u1[j] + dt * r2[j]);
    if ((i + 1) % every === 0 && out.length < nsave) out.push(Float64Array.from(u));
  }
  return out;
}

export const filt = (u: Float64Array) => Float64Array.from({ length: u.length / K }, (_, i) => { let s = 0; for (let k = 0; k < K; k++) s += u[i * K + k]; return s / K; });
export const tauOf = (u: Float64Array) => { const ub = filt(u), u2 = filt(u.map((v) => v * v)); return ub.map((v, i) => 0.5 * (u2[i] - v * v)); };
export const ux = (ub: Float64Array) => { const N = ub.length; return ub.map((_, j) => (ub[(j + 1) % N] - ub[(j - 1 + N) % N]) / (2 * D)); };
const feats = (ub: Float64Array) => { const g = ux(ub); return Array.from(ub, (v, j) => [g[j] ** 2, -Math.abs(g[j]) * g[j], v * g[j], v * v, g[j]]); };

/** 작은 최소제곱 min ‖Aw − y‖ (Householder QR) */
function lstsq(A: number[][], y: number[]) {
  const m = A.length, n = A[0].length;
  const R = A.map((r) => [...r]), b = [...y];
  for (let k = 0; k < n; k++) {
    let nrm = 0;
    for (let i = k; i < m; i++) nrm += R[i][k] ** 2;
    nrm = Math.sqrt(nrm);
    const alpha = R[k][k] > 0 ? -nrm : nrm;
    const v = new Float64Array(m);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] ** 2;
    if (vv === 0) continue;
    for (let j = k; j < n; j++) { let s = 0; for (let i = k; i < m; i++) s += v[i] * R[i][j]; s = (2 * s) / vv; for (let i = k; i < m; i++) R[i][j] -= s * v[i]; }
    let s = 0; for (let i = k; i < m; i++) s += v[i] * b[i]; s = (2 * s) / vv; for (let i = k; i < m; i++) b[i] -= s * v[i];
  }
  const w = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= R[i][j] * w[j]; w[i] = s / R[i][i]; }
  return w;
}

export type Key = "none" | "smag" | "grad" | "lin";
export const NAMES: Record<Key, string> = { none: "모델 없음", smag: "Smagorinsky", grad: "기울기 모델", lin: "선형 회귀(5특징)" };
export type Model = Exclude<Key, "none">;
export const KEYS: Model[] = ["smag", "grad", "lin"];

/** 학습: 시드 0–3의 ν = 0.04 고해상도 궤적. 세 closure를 돌려준다. */
export function train(fine: Float64Array[][]) {
  const Ub: Float64Array[] = [], Y: number[] = [];
  for (let s = 0; s < 4; s++) for (const U of fine[s]) { Ub.push(filt(U)); Y.push(...tauOf(U)); }
  const smagF: number[] = [], gradF: number[] = [], X: number[][] = [];
  for (const ub of Ub) { const g = ux(ub); g.forEach((v) => { smagF.push(-D * D * Math.abs(v) * v); gradF.push(D * D * v * v); }); X.push(...feats(ub)); }
  const fit = (f: number[]) => f.reduce((s, v, i) => s + v * Y[i], 0) / f.reduce((s, v) => s + v * v, 0);
  const Cs = Math.max(fit(smagF), 0), Cg = fit(gradF), w = lstsq(X, Y);
  const models: Record<Key, Closure> = {
    none: null,
    smag: (ub) => { const g = ux(ub); return g.map((v) => -Cs * D * D * Math.abs(v) * v); },
    grad: (ub) => { const g = ux(ub); return g.map((v) => Cg * D * D * v * v); },
    lin: (ub) => Float64Array.from(feats(ub), (f) => f.reduce((s, v, i) => s + v * w[i], 0)),
  };
  return { models, Cs, Cg, w };
}

const norm = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
const meanSq = (a: Float64Array) => a.reduce((s, v) => s + v * v, 0) / a.length;

/** a priori: 시험 궤적(고해상도)을 걸러 τ 예측을 참 τ와 비교 */
export function aPriori(models: Record<Key, Closure>, refs: Float64Array[][]) {
  const Ub: Float64Array[] = [], T: Float64Array[] = [];
  for (const tr of refs) for (const U of tr) { Ub.push(filt(U)); T.push(tauOf(U)); }
  const Pi = (ts: Float64Array[]) => ts.map((t, i) => { const g = ux(Ub[i]); let s = 0; t.forEach((v, j) => (s += v * g[j])); return s * D; });
  const truePi = Pi(T);
  const res = {} as Record<Model, { corr: number; rel: number; pi: number; back: number; pred: Float64Array[] }>;
  const tFlat = T.flatMap((t) => [...t]);
  for (const k of KEYS) {
    const pred = Ub.map((ub) => models[k]!(ub));
    const p = pred.flatMap((t) => [...t]);
    const mp = p.reduce((s, v) => s + v, 0) / p.length, mt = tFlat.reduce((s, v) => s + v, 0) / tFlat.length;
    let sxy = 0, sxx = 0, syy = 0, se = 0;
    p.forEach((v, i) => { sxy += (v - mp) * (tFlat[i] - mt); sxx += (v - mp) ** 2; syy += (tFlat[i] - mt) ** 2; se += (v - tFlat[i]) ** 2; });
    const pis = Pi(pred);
    res[k] = { corr: sxy / Math.sqrt(sxx * syy), rel: Math.sqrt(se) / norm(tFlat), pi: pis.reduce((s, v) => s + v, 0) / pis.length, back: pis.filter((v) => v > 0).length / pis.length, pred };
  }
  return { res, T, truePi: truePi.reduce((s, v) => s + v, 0) / truePi.length, trueBack: truePi.filter((v) => v > 0).length / truePi.length };
}

/** a posteriori: closure를 넣은 32칸 solver를 걸러 낸 초기조건부터 적분, 최종 시각을 비교 */
export function aPosteriori(models: Record<Key, Closure>, nu: number, refs: Map<number, Float64Array[]>) {
  const out = {} as Record<Key, { rel: number; energy: number; final: Float64Array[] }>;
  for (const k of ["none", ...KEYS] as Key[]) {
    const errs: number[] = [], ee: number[] = [], final: Float64Array[] = [];
    for (const [s, U] of refs) {
      const ref = filt(U[U.length - 1]);
      const les = run(filt(ic(s, NF)), D, nu, models[k]);
      const last = les[les.length - 1];
      final.push(last);
      errs.push(norm(last.map((v, j) => v - ref[j])) / norm(ref));
      ee.push(meanSq(last) / meanSq(ref) - 1);
    }
    out[k] = { rel: errs.reduce((a, b) => a + b, 0) / errs.length, energy: ee.reduce((a, b) => a + b, 0) / ee.length, final };
  }
  return out;
}
