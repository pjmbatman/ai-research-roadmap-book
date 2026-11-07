// 26.2장 셀 1–3과 같은 계산: 점성 Burgers 해(유한체적 + Heun), 스냅샷 16개의 POD, 모드 계수의 log ν 3차 다항식 회귀.
export const NX = 128;
const TWO_PI = 2 * Math.PI;

/** u_t + (u²/2)_x = ν u_xx, 주기 경계 [0, 2π), u0 = 0.5 + sin x, 시각 T의 해 (N,) */
export function solve(nu: number, N = NX, T = 1.5): Float64Array {
  const dx = TWO_PI / N;
  const u = new Float64Array(N);
  for (let j = 0; j < N; j++) u[j] = 0.5 + Math.sin(j * dx);
  const n = Math.ceil(T / (0.4 * Math.min(dx / 1.5, (dx * dx) / (2 * nu))));
  const dt = T / n;
  const F = new Float64Array(N), r1 = new Float64Array(N), u1 = new Float64Array(N), r2 = new Float64Array(N);
  const rhs = (v: Float64Array, out: Float64Array) => {
    for (let j = 0; j < N; j++) {
      const a = v[j], b = v[(j + 1) % N];
      const s = Math.max(Math.abs(a), Math.abs(b));
      F[j] = 0.25 * (a * a + b * b) - 0.5 * s * (b - a) - (nu * (b - a)) / dx;
    }
    for (let j = 0; j < N; j++) out[j] = -(F[j] - F[(j - 1 + N) % N]) / dx;
  };
  for (let i = 0; i < n; i++) {
    rhs(u, r1);
    for (let j = 0; j < N; j++) u1[j] = u[j] + dt * r1[j];
    rhs(u1, r2);
    for (let j = 0; j < N; j++) u[j] = 0.5 * (u[j] + u1[j] + dt * r2[j]);
  }
  return u;
}

/** 대칭 행렬의 고유분해(Jacobi). 고윳값 내림차순 */
function eigSym(A: number[][]) {
  const n = A.length;
  const a = A.map((r) => [...r]);
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < n; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < n; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return { val: idx.map((i) => a[i][i]), vec: idx.map((i) => V.map((row) => row[i])) };
}

/** 최소제곱 다항식(차수 deg) 적합. 입력을 표준화해 풀고 예측 함수를 돌려준다(np.polyfit과 같은 다항식). */
function polyfit(x: number[], y: number[], deg: number) {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  const sc = Math.max(...x.map((v) => Math.abs(v - m))) || 1;
  const t = x.map((v) => (v - m) / sc);
  const p = deg + 1;
  const A = Array.from({ length: p }, () => new Array(p + 1).fill(0));
  t.forEach((ti, i) => {
    const pw = Array.from({ length: p }, (_, k) => ti ** k);
    for (let r = 0; r < p; r++) { for (let c = 0; c < p; c++) A[r][c] += pw[r] * pw[c]; A[r][p] += pw[r] * y[i]; }
  });
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r < p; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= p; k++) A[r][k] -= f * A[c][k]; }
  }
  const w = A.map((row, i) => row[p] / row[i]);
  return (v: number) => { const tt = (v - m) / sc; let s = 0; for (let k = p - 1; k >= 0; k--) s = s * tt + w[k]; return s; };
}

export const NU_TRAIN = Array.from({ length: 16 }, (_, i) => 0.05 * Math.pow(4, i / 15)); // np.geomspace(0.05, 0.2, 16)
export const RMAX = 8;

export function buildPOD() {
  const X = NU_TRAIN.map((nu) => solve(nu));
  const M = X.length;
  const mean = new Float64Array(NX);
  X.forEach((u) => u.forEach((v, j) => (mean[j] += v / M)));
  const Xc = X.map((u) => u.map((v, j) => v - mean[j]));
  const G = Xc.map((a) => Xc.map((b) => a.reduce((s, v, j) => s + v * b[j], 0)));
  const { val, vec } = eigSym(G);
  const s2 = val.map((v) => Math.max(v, 0));
  const total = s2.reduce((s, v) => s + v, 0);
  // 오른쪽 특이벡터 φ_k = X_cᵀ u_k / s_k (RMAX개)
  const modes = Array.from({ length: RMAX }, (_, k) => {
    const s = Math.sqrt(s2[k]);
    const phi = new Float64Array(NX);
    for (let i = 0; i < M; i++) for (let j = 0; j < NX; j++) phi[j] += (vec[k][i] * Xc[i][j]) / s;
    return phi;
  });
  const cum = s2.slice(0, RMAX).map((_, k) => s2.slice(0, k + 1).reduce((s, v) => s + v, 0) / total);
  // 버린 에너지 1 − E_r를 뒤쪽 특이값의 합으로 직접 계산한다(1에서 빼는 반올림 오차를 피한다)
  const discard = s2.slice(0, RMAX).map((_, k) => s2.slice(k + 1).reduce((s, v) => s + v, 0) / total);
  const logNu = NU_TRAIN.map(Math.log);
  const coefFns = modes.map((phi) => polyfit(logNu, Xc.map((u) => u.reduce((s, v, j) => s + v * phi[j], 0)), 3));
  return { mean, modes, cum, discard, coefFns };
}
export type POD = ReturnType<typeof buildPOD>;

export function surrogate(P: POD, nu: number, r: number) {
  const u = Float64Array.from(P.mean);
  for (let k = 0; k < r; k++) { const a = P.coefFns[k](Math.log(nu)); const phi = P.modes[k]; for (let j = 0; j < NX; j++) u[j] += a * phi[j]; }
  return u;
}
export function project(P: POD, ref: Float64Array, r: number) {
  const u = Float64Array.from(P.mean);
  for (let k = 0; k < r; k++) {
    const phi = P.modes[k];
    let a = 0;
    for (let j = 0; j < NX; j++) a += (ref[j] - P.mean[j]) * phi[j];
    for (let j = 0; j < NX; j++) u[j] += a * phi[j];
  }
  return u;
}
export const relErr = (a: Float64Array, b: Float64Array) => {
  let n = 0, d = 0;
  for (let j = 0; j < a.length; j++) { n += (a[j] - b[j]) ** 2; d += b[j] ** 2; }
  return Math.sqrt(n / d);
};
/** 정답 없이 계산하는 물리 검사(26.2장 셀 3): 질량, 최대·최소 원리 [-0.5, 1.5], 주기 총변동 ≤ 4 */
export function checks(u: Float64Array) {
  let mass = 0, tv = 0, mx = -Infinity, mn = Infinity;
  for (let j = 0; j < u.length; j++) { mass += u[j]; tv += Math.abs(u[(j + 1) % u.length] - u[j]); mx = Math.max(mx, u[j]); mn = Math.min(mn, u[j]); }
  mass *= TWO_PI / u.length;
  return { massErr: Math.abs(mass - Math.PI), tv, bounds: mx <= 1.5 + 1e-9 && mn >= -0.5 - 1e-9, mx, mn };
}
