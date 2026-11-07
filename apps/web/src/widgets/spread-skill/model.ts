// 27.2장 셀 4와 같은 설정의 앙상블: Lorenz-96(K = 40, F = 8, Δt = 0.05 ≈ 6시간, RK4)의 참 동역학,
// 분석 오차 0.5(분석 = 참값 + N(0, 0.5²)), 멤버 20개 = 분석 + N(0, σ_p²). 본문은 σ_p = 0.5(분석 오차와 같은 크기)다.
// 위젯은 같은 구성을 시드 고정 난수로 다시 만든다(예보 시작 시점 수는 브라우저 계산량에 맞춰 줄였다).
import { rng } from "../frame";

export const K = 40, F = 8, DT = 0.05, LEADS = 40, M = 20, NF = 80, GAP = 10, ANALYSIS_SD = 0.5;

function tend(x: Float64Array, o: number, out: Float64Array, oo: number) {
  for (let i = 0; i < K; i++) {
    const p1 = x[o + ((i + 1) % K)], m2 = x[o + ((i + K - 2) % K)], m1 = x[o + ((i + K - 1) % K)];
    out[oo + i] = (p1 - m2) * m1 - x[o + i] + F;
  }
}
/** n개 상태(각 K)를 RK4 한 step 전진(제자리) */
let buf: Float64Array[] = [];
export function step(x: Float64Array, n: number) {
  if (!buf.length || buf[0].length !== n * K) buf = Array.from({ length: 5 }, () => new Float64Array(n * K));
  const [k1, k2, k3, k4, y] = buf;
  for (let s = 0; s < n; s++) tend(x, s * K, k1, s * K);
  for (let j = 0; j < n * K; j++) y[j] = x[j] + (DT / 2) * k1[j];
  for (let s = 0; s < n; s++) tend(y, s * K, k2, s * K);
  for (let j = 0; j < n * K; j++) y[j] = x[j] + (DT / 2) * k2[j];
  for (let s = 0; s < n; s++) tend(y, s * K, k3, s * K);
  for (let j = 0; j < n * K; j++) y[j] = x[j] + DT * k3[j];
  for (let s = 0; s < n; s++) tend(y, s * K, k4, s * K);
  for (let j = 0; j < n * K; j++) x[j] += (DT / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
}

/** 참 궤적, 분석, 표준정규 섭동 Z(멤버 × 시작 × K). 섭동 크기를 바꿀 때 같은 Z를 다시 쓴다(공통 난수). */
export function setup(seed = 0) {
  const r = rng(seed);
  const x = Float64Array.from({ length: K }, () => F + r.normal());
  for (let i = 0; i < 1000; i++) step(x, 1); // spin-up
  const len = NF * GAP + LEADS + 1;
  const traj: Float64Array[] = [];
  for (let t = 0; t < len; t++) { step(x, 1); traj.push(Float64Array.from(x)); }
  const truth = Array.from({ length: LEADS + 1 }, (_, h) => { const a = new Float64Array(NF * K); for (let s = 0; s < NF; s++) a.set(traj[s * GAP + h], s * K); return a; });
  const analysis = Float64Array.from(truth[0], (v) => v + ANALYSIS_SD * r.normal());
  const Z = Float64Array.from({ length: M * NF * K }, () => r.normal());
  return { truth, analysis, Z };
}
export type Setup = ReturnType<typeof setup>;

/** 섭동 σ_p의 앙상블을 굴려 리드별 spread, 앙상블 평균 RMSE, rank histogram(M+1칸)을 계산한다. */
export function ensemble(S: Setup, sigma: number) {
  const n = M * NF, P = NF * K;
  const x = new Float64Array(n * K);
  for (let m = 0; m < M; m++) for (let j = 0; j < P; j++) x[m * P + j] = S.analysis[j] + sigma * S.Z[m * P + j];
  const spread: number[] = [], rmse: number[] = [], ranks: number[][] = [];
  for (let h = 0; h <= LEADS; h++) {
    if (h > 0) step(x, n);
    const tr = S.truth[h];
    let v = 0, e = 0;
    const hist = new Array(M + 1).fill(0);
    for (let j = 0; j < P; j++) {
      let mean = 0, sq = 0, rank = 0;
      for (let m = 0; m < M; m++) { const a = x[m * P + j]; mean += a; sq += a * a; if (a < tr[j]) rank++; }
      mean /= M;
      v += (sq - M * mean * mean) / (M - 1);
      e += (mean - tr[j]) ** 2;
      hist[rank]++;
    }
    spread.push(Math.sqrt(v / P));
    rmse.push(Math.sqrt(e / P));
    ranks.push(hist.map((k) => k / P));
  }
  return { spread, rmse, ranks };
}
