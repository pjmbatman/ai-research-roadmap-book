// 26.4장 셀 1과 같은 계산: 강철 외팔보(길이 1 m, 끝단 하중 1 kN), 폭 1–5 cm(41칸) × 높이 2–10 cm(81칸) 격자 설계.
// 질량 m = ρ b h L, 끝단 처짐 δ = F L³ / (3 E I) = 4 F L³ / (E b h³), 뿌리 굽힘 응력 σ = 6 F L / (b h²) ≤ 허용 응력.
export const F_N = 1000, L_M = 1, E_PA = 200e9, RHO = 7850;
const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + i * ((b - a) / (n - 1))));
export const BS = linspace(0.01, 0.05, 41), HS = linspace(0.02, 0.1, 81);

export interface Design { b: number; h: number; mass: number; defl: number; sigma: number }
export const evalDesign = (b: number, h: number): Design => ({
  b, h,
  mass: RHO * b * h * L_M,
  defl: ((4 * F_N * L_M ** 3) / (E_PA * b * h ** 3)) * 1e3,
  sigma: (6 * F_N * L_M) / (b * (h * h)),
});
// np.meshgrid(b, h).ravel() 순서: 높이가 바깥, 폭이 안쪽
export const ALL: Design[] = HS.flatMap((h) => BS.map((b) => evalDesign(b, h)));

/** 응력 제약을 만족하는 설계와 그중 Pareto 최적(질량·처짐 모두 최소화) */
export function pareto(sigAllowPa: number) {
  const ok = ALL.filter((d) => d.sigma <= sigAllowPa);
  // 질량 오름차순, 같은 질량이면 처짐 오름차순으로 훑는다. 앞선(더 가벼운) 설계의 최소 처짐보다 엄격히 작고,
  // 같은 질량 무리 안에서 최소 처짐이면 어느 설계에도 지배되지 않는다.
  const idx = ok.map((_, i) => i).sort((i, j) => ok[i].mass - ok[j].mass || ok[i].defl - ok[j].defl);
  const front = new Set<Design>();
  let best = Infinity;
  for (let s = 0; s < idx.length;) {
    let e = s;
    while (e < idx.length && ok[idx[e]].mass === ok[idx[s]].mass) e++;
    const gmin = ok[idx[s]].defl;
    for (let t = s; t < e; t++) { const d = ok[idx[t]]; if (d.defl < best && d.defl === gmin) front.add(d); }
    best = Math.min(best, gmin);
    s = e;
  }
  return { ok, front };
}

/** d를 지배하는 설계 중 질량이 가장 가까운 것(없으면 null) */
export function dominator(d: Design, ok: Design[]) {
  let pick: Design | null = null;
  for (const q of ok) {
    if (q.mass <= d.mass && q.defl <= d.defl && (q.mass < d.mass || q.defl < d.defl)) {
      if (!pick || Math.abs(q.mass - d.mass) + Math.abs(Math.log(q.defl / d.defl)) < Math.abs(pick.mass - d.mass) + Math.abs(Math.log(pick.defl / d.defl))) pick = q;
    }
  }
  return pick;
}
