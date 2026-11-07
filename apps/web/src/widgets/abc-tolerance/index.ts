// ABC 허용 오차 (data/widgets.toml: abc-tolerance). 25.1장의 AR(1) 예: x_t = ρ x_{t−1} + ε_t, T = 50, ρ ~ U(0, 1), 참값 ρ = 0.6.
// 관측 시계열은 장의 코드(np.random.default_rng(1))가 만든 값을 그대로 넣었다(요약 통계 −0.140, 1.093, 0.554, 정확한 사후 0.563 ± 0.134).
// 참조 테이블 2만 개는 위젯 안에서 시드 고정 난수로 만든다(장의 표와 난수는 다르다).
// 거리는 각 요약 통계를 표준편차로 나눈 유클리드 거리, 수용 비율 q의 분위수가 허용 오차 ε이다(장의 abc 함수와 같다).
// 위: 결합 표본 (ρ, 요약 통계)과 관측값 근처의 띠(셋 다 쓰면 거리 d). 아래: 수용된 ρ의 히스토그램, 정확한 사후분포, 사전분포.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";

export const T = 50, NSIM = 20000;
export const X_OBS = [0.0, 0.345584, 1.028969, 0.947818, -0.734466, 0.464676, 0.72518, -0.101845, 0.520011, 0.676579, 0.70008, 0.44847, 0.815795, -0.246977, -0.311096, -0.668777, 0.19758, 0.15827, -0.197495, -0.900405, -0.797435, -0.470319, -0.557794, 0.959387, 1.582357, -1.761748, -2.946062, -1.942409, -1.587636, -0.738939, -0.226041, 1.982214, 0.077308, -0.33122, 1.844039, 1.753127, 1.714939, 0.514957, -1.339101, -0.635996, -0.272583, -1.390902, -1.517768, -0.982704, -1.534374, -1.018895, -0.515854, -0.273926, -0.670647, 0.19136];

/** 평균, 분산(n으로 나눔), lag-1 자기상관 */
export function summaries(x: ArrayLike<number>): [number, number, number] {
  const n = x.length;
  let m = 0; for (let t = 0; t < n; t++) m += x[t]; m /= n;
  let v = 0, c = 0;
  for (let t = 0; t < n; t++) { const a = x[t] - m; v += a * a; if (t > 0) c += a * (x[t - 1] - m); }
  return [m, v / n, c / v];
}

export function table(seed = 11) {
  const r = rng(seed);
  const rho = new Float64Array(NSIM), S = [new Float64Array(NSIM), new Float64Array(NSIM), new Float64Array(NSIM)];
  const x = new Float64Array(T);
  for (let i = 0; i < NSIM; i++) {
    const p = r.u(); rho[i] = p;
    x[0] = 0;
    for (let t = 1; t < T; t++) x[t] = p * x[t - 1] + r.normal();
    const s = summaries(x);
    S[0][i] = s[0]; S[1][i] = s[1]; S[2][i] = s[2];
  }
  return { rho, S };
}

/** 격자 위의 정확한 사후분포(채점용 정답) */
export function exactPosterior() {
  const G = 2001, grid = Array.from({ length: G }, (_, i) => i / (G - 1));
  const ll = grid.map((g) => { let s = 0; for (let t = 1; t < T; t++) s += (X_OBS[t] - g * X_OBS[t - 1]) ** 2; return -0.5 * s; });
  const mx = Math.max(...ll);
  const p = ll.map((v) => Math.exp(v - mx)); const z = p.reduce((a, b) => a + b, 0);
  const post = p.map((v) => v / z);
  const mean = grid.reduce((s, g, i) => s + g * post[i], 0);
  const sd = Math.sqrt(grid.reduce((s, g, i) => s + (g - mean) ** 2 * post[i], 0));
  return { grid, dens: post.map((v) => v * (G - 1)), mean, sd };
}

export type Stat = "mean" | "var" | "ac" | "all";
export const COLS: Record<Stat, number[]> = { mean: [0], var: [1], ac: [2], all: [0, 1, 2] };

/** 거리와 정렬된 거리. scale은 참조 테이블의 표준편차(np.std, n으로 나눔). */
export function distances(tab: ReturnType<typeof table>, cols: number[]) {
  const sObs = summaries(X_OBS);
  const scale = cols.map((c) => { const a = tab.S[c]; let m = 0; for (const v of a) m += v; m /= a.length; let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / a.length); });
  const d = new Float64Array(NSIM);
  for (let i = 0; i < NSIM; i++) { let s = 0; cols.forEach((c, k) => { s += ((tab.S[c][i] - sObs[c]) / scale[k]) ** 2; }); d[i] = Math.sqrt(s); }
  const sorted = Float64Array.from(d).sort();
  return { d, sorted, scale, sObs };
}
/** np.quantile(d, q) (선형 보간) */
export const quantile = (sorted: Float64Array, q: number) => { const h = q * (sorted.length - 1), lo = Math.floor(h); return lo + 1 < sorted.length ? sorted[lo] + (h - lo) * (sorted[lo + 1] - sorted[lo]) : sorted[lo]; };

export function accept(tab: ReturnType<typeof table>, dist: ReturnType<typeof distances>, q: number) {
  const eps = quantile(dist.sorted, q);
  let n = 0, s = 0, s2 = 0;
  for (let i = 0; i < NSIM; i++) if (dist.d[i] <= eps) { n++; s += tab.rho[i]; s2 += tab.rho[i] ** 2; }
  const mean = s / n;
  return { eps, n, mean, sd: Math.sqrt(Math.max(0, s2 / n - mean * mean)) };
}

const NAMES: Record<Stat, string> = { mean: "평균", var: "분산", ac: "자기상관", all: "셋 다" };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "ABC 허용 오차", "manipulate", narrow ? 1.05 : 0.62, "사전분포에서 뽑은 ρ와 요약 통계의 결합 표본, 관측값 근처의 수용 띠, 수용된 ρ의 히스토그램과 정확한 사후분포");
  const tab = table();
  const ex = exactPosterior();
  const cache = new Map<Stat, ReturnType<typeof distances>>();
  const dist = (s: Stat) => { if (!cache.has(s)) cache.set(s, distances(tab, COLS[s])); return cache.get(s)!; };
  let stat: Stat = "ac", lq = -2;
  let acc = accept(tab, dist(stat), 10 ** lq);

  // 단일 통계의 그림 범위(0.5–99.5% 분위수)
  const range = (c: number) => { const a = Float64Array.from(tab.S[c]).sort(); return [a[Math.floor(0.005 * NSIM)], a[Math.floor(0.995 * NSIM)]] as [number, number]; };
  const ranges = [range(0), range(1), range(2)];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const left = 42, right = W - 10, top = 10;
    const split = Math.round(H * 0.56), hb = H - 22;
    const X = (r: number) => left + r * (right - left);
    const D = dist(stat), single = COLS[stat].length === 1, c = COLS[stat][0];
    let y0: number, y1: number;
    if (single) [y0, y1] = ranges[c]; else { y0 = 0; y1 = quantile(D.sorted, 0.995); }
    const pad = (y1 - y0) * 0.04; y0 -= single ? pad : 0; y1 += pad;
    const Y = (v: number) => top + ((y1 - v) / (y1 - y0)) * (split - top);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
    ctx.strokeStyle = C.faint; ctx.strokeRect(left, top, right - left, split - top);
    // 수용 띠
    const lo = single ? D.sObs[c] - acc.eps * D.scale[0] : 0, hi = single ? D.sObs[c] + acc.eps * D.scale[0] : acc.eps;
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, split - top); ctx.clip();
    ctx.fillStyle = C.fill; ctx.fillRect(left, Y(hi) - 1, right - left, Math.max(3, Y(lo) - Y(hi) + 2));
    // 결합 표본
    const val = single ? tab.S[c] : D.d;
    ctx.fillStyle = alpha(C.mid, 0.35);
    for (let i = 0; i < NSIM; i++) if (D.d[i] > acc.eps) ctx.fillRect(X(tab.rho[i]) - 0.75, Y(val[i]) - 0.75, 1.5, 1.5);
    ctx.fillStyle = C.edge;
    for (let i = 0; i < NSIM; i++) if (D.d[i] <= acc.eps) ctx.fillRect(X(tab.rho[i]) - 1.25, Y(val[i]) - 1.25, 2.5, 2.5);
    if (single) { ctx.strokeStyle = C.ink; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(left, Y(D.sObs[c])); ctx.lineTo(right, Y(D.sObs[c])); ctx.stroke(); ctx.setLineDash([]); }
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    const ticks = single ? [y0 + pad, (y0 + y1) / 2, y1 - pad] : [0, y1 / 2];
    for (const t of ticks) ctx.fillText(Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(1), left - 5, Y(t) + 4);
    ctx.textAlign = "start";
    const yl = single ? `s = ${NAMES[stat]}` : "거리 d";
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(left + 1, top + 1, ctx.measureText(yl).width + 12, 18);
    ctx.fillStyle = C.ink; ctx.fillText(yl, left + 6, top + 14);
    // 아래: 히스토그램
    const B = 25, cnt = new Array(B).fill(0);
    for (let i = 0; i < NSIM; i++) if (D.d[i] <= acc.eps) cnt[Math.min(B - 1, Math.floor(tab.rho[i] * B))]++;
    const dens = cnt.map((k) => (k / acc.n) * B);
    const hmax = Math.max(4, ...dens, ...ex.dens) * 1.08;
    const bt = split + 24;
    const HY = (v: number) => hb - (v / hmax) * (hb - bt);
    ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(left, hb); ctx.lineTo(right, hb); ctx.stroke();
    ctx.fillStyle = C.edgeSoft;
    dens.forEach((v, b) => ctx.fillRect(X(b / B) + 0.5, HY(v), (right - left) / B - 1, hb - HY(v)));
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(left, HY(1)); ctx.lineTo(right, HY(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ex.grid.forEach((g, i) => { if (i % 5) return; i ? ctx.lineTo(X(g), HY(ex.dens[i])) : ctx.moveTo(X(g), HY(ex.dens[i])); }); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const v of [0, 0.5, 1]) ctx.fillText(String(v), X(v), hb + 13);
    ctx.fillText("ρ", X(0.75), hb + 13);
    ctx.textAlign = "start";
    // 범례
    const items: [string, (x: number, y: number) => void][] = [
      ["수용된 ρ", (x, y) => { ctx.fillStyle = C.edgeSoft; ctx.fillRect(x, y - 5, 14, 10); }],
      ["정확한 사후", (x, y) => { ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); ctx.lineWidth = 1; }],
      ["사전", (x, y) => { ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); ctx.setLineDash([]); }],
    ];
    let lx = left + 6;
    for (const [t, mark] of items) { mark(lx, bt + 2); ctx.fillStyle = C.ink; ctx.fillText(t, lx + 18, bt + 6); lx += ctx.measureText(t).width + 26; }
  }

  function update() {
    acc = accept(tab, dist(stat), 10 ** lq);
    ro.set("n", `수용 ${acc.n}개 / ${NSIM}`);
    ro.set("eps", `ε = ${acc.eps.toPrecision(2)}`);
    const off = Math.abs(acc.mean - ex.mean) > 0.05 || acc.sd > ex.sd * 1.3;
    ro.set("post", `ABC ${acc.mean.toFixed(3)} ± ${acc.sd.toFixed(3)}`, off ? "hot" : "ok");
    ro.set("exact", `정확 ${ex.mean.toFixed(3)} ± ${ex.sd.toFixed(3)}`);
    draw();
  }

  const sp = presets<Stat>((Object.keys(NAMES) as Stat[]).map((k) => ({ key: k, label: NAMES[k] })), (k) => { stat = k; update(); }, stat);
  sp.row.classList.add("segmented");
  const pq = param({ label: "q", min: -3, max: 0, step: 0.05, value: lq, fmt: (v) => { const q = 10 ** v; return q >= 0.1 ? q.toFixed(2) : q >= 0.01 ? q.toFixed(3) : q.toFixed(4); }, aria: "수용 비율 q(로그 눈금): 거리가 가장 가까운 비율 q의 시뮬레이션을 받아들인다", onInput: (v) => { lq = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pq.row);
  const ro = readouts();
  f.panel.append(sp.row, grid, ro.row);
  f.onResize(draw);
  update();
}
