// 신뢰구간의 포함률 (data/widgets.toml: ci-coverage).
// 12.4장의 예: 크기 n인 표본으로 t 구간 X̄ ± t*·S/√n (t* = t_{n−1}의 1 − α/2 분위수)을 만들고, 같은 실험을 반복해
// 구간이 참 평균을 덮는 비율(포함률)을 잰다. 데이터는 정규 N(5, 2²) 또는 로그정규 LogNormal(0, 1)(참 평균 e^{1/2}).
// 실험 R번의 표본은 고정 시드 표준정규 z로 만든다(정규: 5 + 2z, 로그정규: e^z). 그림은 처음 100번의 구간이다.
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const R = 4000, NMAX = 100, SHOW = 100;
const Zs = (() => { const r = rng(1979); const z = new Float64Array(R * NMAX); for (let i = 0; i < z.length; i++) z[i] = r.normal(); return z; })();

const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(a: number, b: number, x: number) {
  const qab = a + b, qap = a + 1, qam = a - 1, tiny = 1e-300;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function betaI(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
/** 자유도 df인 t 분포의 cdf (t ≥ 0) */
const tCdf = (t: number, df: number) => 1 - 0.5 * betaI(df / 2, 0.5, df / (df + t * t));
/** t_df의 q 분위수 (q ≥ 0.5), 이분법 */
export function tQuantile(q: number, df: number) {
  let lo = 0, hi = 1;
  while (tCdf(hi, df) < q) hi *= 2;
  for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (tCdf(m, df) < q) lo = m; else hi = m; }
  return (lo + hi) / 2;
}

export const DISTS = {
  normal: { label: "정규 N(5, 2²)", mean: 5, sd: 2, gen: (z: number) => 5 + 2 * z },
  lognormal: { label: "로그정규 (꼬리 김)", mean: Math.exp(0.5), sd: Math.sqrt((Math.E - 1) * Math.E), gen: (z: number) => Math.exp(z) },
} as const;
export type DK = keyof typeof DISTS;

/** 실험 R번의 구간과 포함 여부 */
export function intervals(dist: DK, n: number, level: number) {
  const D = DISTS[dist], ts = tQuantile(1 - (1 - level) / 2, n - 1);
  const lo = new Float64Array(R), hi = new Float64Array(R), mean = new Float64Array(R);
  let cover = 0, missLow = 0, missHigh = 0;
  for (let r = 0; r < R; r++) {
    let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) { const v = D.gen(Zs[r * NMAX + i]); s += v; s2 += v * v; }
    const m = s / n, sd = Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1))), half = (ts * sd) / Math.sqrt(n);
    mean[r] = m; lo[r] = m - half; hi[r] = m + half;
    if (hi[r] < D.mean) missLow++; else if (lo[r] > D.mean) missHigh++; else cover++;
  }
  return { lo, hi, mean, ts, coverage: cover / R, missLow: missLow / R, missHigh: missHigh / R };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "신뢰구간 포함률", "manipulate", 0.62, "같은 실험을 100번 반복해 만든 신뢰구간과 참 평균, 참 평균을 놓친 구간 강조");
  let dist: DK = (attrs.dist in DISTS ? attrs.dist : "lognormal") as DK;
  let n = 10, level = 0.95;
  let res = intervals(dist, n, level);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const D = DISTS[dist];
    const PL = 12, PR = 12, top = 22, bottom = H - 24;
    const a = D.mean - 2 * D.sd, b = D.mean + 2 * D.sd;
    const X = (v: number) => PL + ((v - a) / (b - a)) * (W - PL - PR);
    const rowH = (bottom - top) / SHOW;
    // 눈금
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    const stepT = D.sd >= 2 ? 2 : 1;
    for (let v = Math.ceil(a / stepT) * stepT; v <= b; v += stepT) {
      ctx.beginPath(); ctx.moveTo(X(v), top); ctx.lineTo(X(v), bottom); ctx.stroke();
      const s = String(v);
      ctx.fillText(s, X(v) - ctx.measureText(s).width / 2, bottom + 15);
    }
    // 구간
    const lw = Math.max(1, Math.min(2.2, rowH * 0.6));
    for (let r = 0; r < SHOW; r++) {
      const y = top + (r + 0.5) * rowH;
      const miss = res.hi[r] < D.mean || res.lo[r] > D.mean;
      ctx.strokeStyle = miss ? C.edge : alpha(C.ink2, 0.6);
      ctx.lineWidth = miss ? lw + 0.6 : lw;
      const x0 = X(Math.max(a, res.lo[r])), x1 = X(Math.min(b, res.hi[r]));
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      // 그림 밖으로 나간 끝은 화살촉으로
      ctx.fillStyle = miss ? C.edge : alpha(C.ink2, 0.6);
      if (res.lo[r] < a) { ctx.beginPath(); ctx.moveTo(x0 - 3, y); ctx.lineTo(x0 + 2, y - 2.5); ctx.lineTo(x0 + 2, y + 2.5); ctx.fill(); }
      if (res.hi[r] > b) { ctx.beginPath(); ctx.moveTo(x1 + 3, y); ctx.lineTo(x1 - 2, y - 2.5); ctx.lineTo(x1 - 2, y + 2.5); ctx.fill(); }
      ctx.fillStyle = C.ink;
      const mx = X(res.mean[r]);
      if (mx >= PL && mx <= W - PR) ctx.fillRect(mx - 1, y - lw / 2 - 0.5, 2, lw + 1);
    }
    // 참 평균
    ctx.strokeStyle = C.good; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(D.mean), top - 6); ctx.lineTo(X(D.mean), bottom); ctx.stroke();
    ctx.fillStyle = C.good;
    const lab = `참 평균 ${D.mean.toFixed(dist === "normal" ? 0 : 3)}`;
    ctx.fillText(lab, Math.min(W - PR - ctx.measureText(lab).width, X(D.mean) + 5), top - 8);
    ctx.fillStyle = C.mid;
    ctx.fillText(`실험 ${SHOW}번`, PL, top - 8);
  }

  let raf = 0;
  function update() {
    res = intervals(dist, n, level);
    let shownMiss = 0;
    const D = DISTS[dist];
    for (let r = 0; r < SHOW; r++) if (res.hi[r] < D.mean || res.lo[r] > D.mean) shownMiss++;
    const gap = res.coverage - level;
    ro.set("shown", `그림의 ${SHOW}개 중 ${SHOW - shownMiss}개가 덮음`);
    ro.set("cov", `포함률 ${res.coverage.toFixed(3)} (${R}번, 목표 ${level.toFixed(2)})`, Math.abs(gap) < 0.015 ? "ok" : "hot");
    ro.set("miss", `놓침: 구간이 아래 ${(res.missLow * 100).toFixed(1)}% · 위 ${(res.missHigh * 100).toFixed(1)}%`);
    ro.set("t", `t* = ${res.ts.toFixed(3)} (자유도 ${n - 1})`);
    pageState.set("coverage", { dist, n, level, coverage: res.coverage }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets((Object.keys(DISTS) as DK[]).map((k) => ({ key: k, label: DISTS[k].label })), (k) => { dist = k; update(); }, dist);
  pre.row.classList.add("segmented");
  const pn = param({ label: "n", min: 2, max: NMAX, step: 1, value: n, fmt: (v) => String(v), aria: "표본 크기 n", onInput: (v) => { n = v; update(); } });
  const pl = param({ label: "1−α", min: 0.8, max: 0.99, step: 0.01, value: level, fmt: (v) => v.toFixed(2), aria: "신뢰수준 1 − α", onInput: (v) => { level = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row, pl.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
