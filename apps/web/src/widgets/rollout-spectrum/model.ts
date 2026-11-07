// 26.5장 셀 1–3과 같은 계산: 64칸 보존형 Burgers solver(모델 1 step = solver 4 step, Δt = 0.02)로 학습 궤적 36개를 만들고,
// A 직접 갱신(5점 스텐실의 1·2차 항 + ν×2계 차분, 21개)과 B 보존형 유속(3개)을 최소제곱으로 맞춘 뒤 시험 궤적(시드 100–105)에서 굴린다.
import { BENCH_IC } from "./data";

export const N = 64, L = 2 * Math.PI, DT = 0.02, T = 50, dx = L / N;
const at = (u: Float64Array, j: number) => u[((j % N) + N) % N];

/** 시드 행 r(0–11 학습, 12–17 시험)의 초기 장 */
export function ic(r: number, amp = 1) {
  const c = BENCH_IC[r];
  return Float64Array.from({ length: N }, (_, j) => {
    const x = j * dx;
    let v = 0;
    for (let k = 1; k <= 3; k++) v += (c[2 * k - 1] / k) * Math.sin(k * x + c[2 * k]);
    return c[0] + amp * v;
  });
}

export function stepRef(u0: Float64Array, nu: number, sub = 4) {
  let u = Float64Array.from(u0);
  const F = new Float64Array(N);
  for (let s = 0; s < sub; s++) {
    for (let j = 0; j < N; j++) { const a = u[j], b = at(u, j + 1); F[j] = 0.25 * (a * a + b * b) - 0.5 * Math.max(Math.abs(a), Math.abs(b)) * (b - a) - (nu * (b - a)) / dx; }
    u = u.map((v, j) => v - (DT / sub / dx) * (F[j] - at(F, j - 1)));
  }
  return u;
}
export function traj(r: number, nu: number, amp = 1) {
  const out = [ic(r, amp)];
  for (let i = 0; i < T; i++) out.push(stepRef(out[i], nu));
  return out;
}

const featA = (u: Float64Array, nu: number, j: number, out: Float64Array) => {
  const S = [at(u, j - 2), at(u, j - 1), u[j], at(u, j + 1), at(u, j + 2)];
  let k = 0;
  for (let i = 0; i < 5; i++) out[k++] = S[i];
  for (let i = 0; i < 5; i++) for (let m = i; m < 5; m++) out[k++] = S[i] * S[m];
  out[k] = nu * (S[3] - 2 * S[2] + S[1]);
};
const fluxB = (u: Float64Array, nu: number, j: number): [number, number, number] => {
  const a = u[j], b = at(u, j + 1);
  return [(a * a + a * b + b * b) / 6, -nu * (b - a), -Math.abs(a + b) * (b - a)];
};

/** 정규방정식(열 표준화 + 부분 피벗 소거)으로 푸는 최소제곱. 특징 수가 작고 조건수가 10⁴ 수준이라 충분히 정확하다. */
function solveNormal(G: number[][], h: number[]) {
  const n = h.length, sc = G.map((r, i) => Math.sqrt(r[i]) || 1);
  const A = G.map((r, i) => [...r.map((v, j) => v / (sc[i] * sc[j])), h[i] / sc[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < n; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]; }
  }
  const w = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = A[i][n]; for (let j = i + 1; j < n; j++) s -= A[i][j] * w[j]; w[i] = s / A[i][i]; }
  return w.map((v, i) => v / sc[i]);
}

export type Model = (u: Float64Array, nu: number) => Float64Array;
export function fit() {
  const GA = Array.from({ length: 21 }, () => new Array(21).fill(0)), hA = new Array(21).fill(0);
  const GB = Array.from({ length: 3 }, () => new Array(3).fill(0)), hB = new Array(3).fill(0);
  const fa = new Float64Array(21);
  for (let s = 0; s < 12; s++) for (const nu of [0.03, 0.05, 0.1]) {
    const tr = traj(s, nu);
    for (let i = 0; i < T; i++) {
      const x = tr[i], y = tr[i + 1];
      for (let j = 0; j < N; j++) {
        featA(x, nu, j, fa);
        const t = y[j] - x[j];
        for (let a = 0; a < 21; a++) { hA[a] += fa[a] * t; for (let b = a; b < 21; b++) GA[a][b] += fa[a] * fa[b]; }
        const f1 = fluxB(x, nu, j), f0 = fluxB(x, nu, j - 1 + (j === 0 ? N : 0));
        const d = [f1[0] - f0[0], f1[1] - f0[1], f1[2] - f0[2]], tb = x[j] - y[j];
        for (let a = 0; a < 3; a++) { hB[a] += d[a] * tb; for (let b = a; b < 3; b++) GB[a][b] += d[a] * d[b]; }
      }
    }
  }
  for (let a = 0; a < 21; a++) for (let b = 0; b < a; b++) GA[a][b] = GA[b][a];
  for (let a = 0; a < 3; a++) for (let b = 0; b < a; b++) GB[a][b] = GB[b][a];
  const wA = solveNormal(GA, hA), wB = solveNormal(GB, hB);
  const A: Model = (u, nu) => Float64Array.from({ length: N }, (_, j) => { featA(u, nu, j, fa); let s = u[j]; for (let a = 0; a < 21; a++) s += fa[a] * wA[a]; return s; });
  const B: Model = (u, nu) => {
    const F = Float64Array.from({ length: N }, (_, j) => { const f = fluxB(u, nu, j); return f[0] * wB[0] + f[1] * wB[1] + f[2] * wB[2]; });
    return u.map((v, j) => v - (F[j] - at(F, j - 1)));
  };
  return { A, B, wA, wB };
}

/** 에너지 스펙트럼 |û_k|², k = 0..N/2 (rfft) */
export function spectrum(u: Float64Array) {
  return Array.from({ length: N / 2 + 1 }, (_, k) => {
    let re = 0, im = 0;
    for (let j = 0; j < N; j++) { const a = (-2 * Math.PI * k * j) / N; re += u[j] * Math.cos(a); im += u[j] * Math.sin(a); }
    return re * re + im * im;
  });
}
/** 파수 8 이상이 담은 에너지 비율(26.5장의 hi_k) */
export const hiK = (E: number[]) => E.slice(8).reduce((s, v) => s + v, 0) / E.slice(1).reduce((s, v) => s + v, 0);
export const rel = (a: Float64Array, b: Float64Array) => { let n = 0, d = 0; for (let j = 0; j < N; j++) { n += (a[j] - b[j]) ** 2; d += b[j] ** 2; } return Math.sqrt(n / d); };

/** 시험 궤적 하나에서 rollout: 발산(비유한 또는 |u| > 1e3) 직전까지의 상태 */
export function rollout(f: Model, u0: Float64Array, nu: number) {
  const out = [u0];
  for (let i = 0; i < T; i++) {
    const u = f(out[i], nu);
    if (!u.every(Number.isFinite) || Math.max(...u.map(Math.abs)) > 1e3) return { states: out, blown: i + 1 };
    out.push(u);
  }
  return { states: out, blown: 0 };
}
