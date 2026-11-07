// 표집 설계 커버리지 (data/widgets.toml: doe-coverage). 15.6장의 실험 설계 절과 같은 두 파라미터(감쇠비 ζ ∈ [0.05, 0.5], 고유진동수 ω ∈ [1, 4] rad/s).
// 단위 정사각형 [0, 1)²에서 n개를 고른 뒤 범위로 옮긴다. grid는 축마다 k = ⌊√n⌋단계(k²개), random은 균등 난수,
// LHS는 축마다 n구간에 한 점씩(구간 안 위치는 무작위), Sobol은 2차원 Sobol 수열에 무작위 digital shift를 더한 것이다.
// 읽음값: ζ축 투영이 n칸 중 몇 칸을 덮는가(장의 코드와 같은 셈), centered L2-discrepancy(scipy qmc.discrepancy 기본값과 같은 식),
// 그리고 정사각형 안에서 가장 가까운 점까지의 거리가 가장 먼 곳(가장 큰 빈 원의 반지름).
import { frame, param, presets, readouts, el, C, rng } from "../frame";

export type Design = "grid" | "random" | "lhs" | "sobol";
const Z: [number, number] = [0.05, 0.5], OM: [number, number] = [1, 4];

/** 2차원 Sobol 수열(첫 축은 밑 2 van der Corput, 둘째 축은 원시다항식 x + 1, m₁ = 1)에 digital shift를 XOR한다. */
export function sobol2(n: number, shift: [number, number] = [0, 0]): number[][] {
  const v1 = Array.from({ length: 32 }, (_, k) => 2 ** (31 - k));
  const v2: number[] = [2 ** 31];
  for (let k = 1; k < 32; k++) v2.push((v2[k - 1] ^ (v2[k - 1] >>> 1)) >>> 0);
  let a = 0, b = 0;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push([((a ^ shift[0]) >>> 0) / 2 ** 32, ((b ^ shift[1]) >>> 0) / 2 ** 32]);
    let c = 0, m = i; while (m & 1) { m >>= 1; c++; }   // i의 가장 낮은 0 비트 위치(Gray 코드 순서)
    a = (a ^ v1[c]) >>> 0; b = (b ^ v2[c]) >>> 0;
  }
  return out;
}

export function design(kind: Design, n: number, seed: number): number[][] {
  const r = rng(seed * 131 + { grid: 1, random: 2, lhs: 3, sobol: 4 }[kind]);
  if (kind === "grid") {
    const k = Math.max(1, Math.floor(Math.sqrt(n)));
    const g = Array.from({ length: k }, (_, i) => (i + 0.5) / k);
    return g.flatMap((a) => g.map((b) => [a, b]));
  }
  if (kind === "random") return Array.from({ length: n }, () => [r.u(), r.u()]);
  if (kind === "lhs") {
    const perm = () => { const p = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; };
    const px = perm(), py = perm();
    return Array.from({ length: n }, (_, i) => [(px[i] + r.u()) / n, (py[i] + r.u()) / n]);
  }
  return sobol2(n, [Math.floor(r.u() * 2 ** 32), Math.floor(r.u() * 2 ** 32)]);
}

/** centered L2-discrepancy의 제곱(Hickernell, 1998). scipy.stats.qmc.discrepancy(method="CD")가 돌려주는 값과 같은 식이다. */
export function cd2(S: number[][]): number {
  const n = S.length, d = S[0].length;
  let s1 = 0, s2 = 0;
  for (const x of S) { let p = 1; for (let k = 0; k < d; k++) { const a = Math.abs(x[k] - 0.5); p *= 1 + 0.5 * a - 0.5 * a * a; } s1 += p; }
  for (const x of S) for (const y of S) {
    let p = 1;
    for (let k = 0; k < d; k++) p *= 1 + 0.5 * Math.abs(x[k] - 0.5) + 0.5 * Math.abs(y[k] - 0.5) - 0.5 * Math.abs(x[k] - y[k]);
    s2 += p;
  }
  return (13 / 12) ** d - (2 / n) * s1 + s2 / (n * n);
}
/** ζ축 투영이 n칸 중 덮는 칸 수(장의 np.unique(np.floor(S[:, 0] * n))와 같다) */
export const cover = (S: number[][]) => new Set(S.map((x) => Math.floor(x[0] * S.length))).size;
/** 가장 큰 빈 원: 정사각형 안 격자점 중 가장 가까운 설계점까지의 거리가 가장 먼 곳 */
export function emptiest(S: number[][], R = 120) {
  let best = { r: -1, x: 0, y: 0 };
  for (let j = 0; j <= R; j++) for (let i = 0; i <= R; i++) {
    const x = i / R, y = j / R;
    let m = Infinity;
    for (const p of S) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < m) m = d; }
    if (m > best.r) best = { r: m, x, y };
  }
  return { ...best, r: Math.sqrt(best.r) };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "표집 설계 커버리지", "manipulate", narrow ? 1.0 : 0.62, "감쇠비와 고유진동수 평면의 설계점, 축 투영, 가장 큰 빈 원");
  let kind: Design = "random", n = 16, seed = 0;
  let S: number[][] = [];
  let hole = { r: 0, x: 0, y: 0 };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const side = Math.min(H - 58, W - 76);
    const ox = Math.round((W - side) / 2 + 14), oy = 14;
    const PX = (u: number) => ox + u * side, PY = (v: number) => oy + side - v * side;
    // 축마다 n칸(점이 많으면 생략)
    const m = S.length;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    if (m <= 40) {
      ctx.beginPath();
      for (let k = 1; k < m; k++) { ctx.moveTo(PX(k / m), oy); ctx.lineTo(PX(k / m), oy + side); }
      ctx.globalAlpha = 0.45; ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.strokeRect(ox, oy, side, side);
    // 투영(러그): 아래 ζ축, 왼쪽 ω축
    const rug = oy + side + 4;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.4;
    ctx.beginPath();
    S.forEach(([u, v]) => { ctx.moveTo(PX(u), rug); ctx.lineTo(PX(u), rug + 9); ctx.moveTo(ox - 4, PY(v)); ctx.lineTo(ox - 13, PY(v)); });
    ctx.stroke();
    // 가장 큰 빈 원
    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
    ctx.fillStyle = C.fill;
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(PX(hole.x), PY(hole.y), hole.r * side, 0, 7); ctx.fill(); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(PX(hole.x), PY(hole.y), 2.5, 0, 7); ctx.fill();
    // 점
    ctx.fillStyle = C.ink;
    S.forEach(([u, v]) => { ctx.beginPath(); ctx.arc(PX(u), PY(v), m > 64 ? 2.4 : 3.6, 0, 7); ctx.fill(); });
    // 축 이름과 범위
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    const lab = rug + 22;
    ctx.fillText(String(Z[0]), PX(0), lab);
    ctx.fillText(String(Z[1]), PX(1) - ctx.measureText(String(Z[1])).width, lab);
    ctx.fillText("ζ", PX(0.5) - 3, lab);
    ctx.fillText(String(OM[1]), ox - 16 - ctx.measureText(String(OM[1])).width, PY(1) + 8);
    ctx.fillText(String(OM[0]), ox - 16 - ctx.measureText(String(OM[0])).width, PY(0));
    ctx.fillText("ω", ox - 28, PY(0.5) + 4);
  }

  function update() {
    S = design(kind, n, seed);
    hole = emptiest(S);
    const m = S.length;
    ro.set("n", kind === "grid" ? `실행 ${m}개 (${Math.sqrt(m)}×${Math.sqrt(m)})` : `실행 ${m}개`);
    const c = cover(S);
    ro.set("cover", `ζ축 ${m}칸 중 ${c}칸`, c === m ? "ok" : "hot");
    ro.set("cd", `discrepancy ${cd2(S).toFixed(4)}`);
    ro.set("hole", `빈 원 r ${hole.r.toFixed(3)}`);
    draw();
  }

  const kinds = presets<Design>([{ key: "grid", label: "grid" }, { key: "random", label: "random" }, { key: "lhs", label: "LHS" }, { key: "sobol", label: "Sobol" }], (k) => { kind = k; update(); }, kind);
  kinds.row.classList.add("segmented");
  const pn = param({ label: "n", min: 4, max: 128, step: 1, value: n, fmt: (v) => `${v}`, aria: "설계점 수 n", onInput: (v) => { n = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const tools = el("div");
  tools.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center";
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 표본";
  reseed.addEventListener("click", () => { seed++; update(); });
  const btnRow = el("div", "wframe-row");
  btnRow.append(reseed);
  tools.append(kinds.row, btnRow);
  const ro = readouts();
  f.panel.append(tools, grid, ro.row);
  f.onResize(draw);
  update();
}
