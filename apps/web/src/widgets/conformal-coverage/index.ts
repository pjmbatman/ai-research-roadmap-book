// split conformal 포함률 (data/widgets.toml: conformal-coverage).
// 21.3장의 예: y = sin 2x + ε, ε ~ N(0, (0.05 + 0.2x)²), x ~ U(0, 3). 학습 표본 500개로 맞춘 5차 다항식 μ̂(고정).
// 보정 세트 n개를 R = 1000번 새로 뽑아, 각 세트의 q̂(점수 |y − μ̂(x)|의 k = ⌈(n+1)(1−α)⌉번째 값)로 만든 구간의
// 참 포함률 P(|Y − μ̂(X)| ≤ q̂)를 계산한다(x 표본 2000개 위의 정규 cdf 평균, 장과 같은 방법; q 격자에서 선형 보간).
// 히스토그램을 이론 분포 Beta(k, n + 1 − k)와 비교한다. k > n이면 q̂ = ∞(구간이 무한)다.
import { frame, param, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

export const REPS = 1000, NS = [5, 9, 10, 19, 30, 50, 100, 200, 500, 1000];
const f = (x: number) => Math.sin(2 * x), sdTrue = (x: number) => 0.05 + 0.2 * x;

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
/** 정규화된 불완전 베타 함수 I_x(a, b) = Beta(a, b)의 cdf */
export function betaI(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
const betaPdf = (a: number, b: number, x: number) => (x <= 0 || x >= 1 ? 0 : Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)));
/** 표준정규 cdf (erfc의 Chebyshev 근사, 상대오차 < 1.2e−7) */
function Phi(z: number) {
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.5 * x);
  const erfc = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return z >= 0 ? 1 - erfc / 2 : erfc / 2;
}

/** 5차 다항식 최소제곱(정규방정식, x를 [−1, 1]로 옮겨 푼다) */
function polyfit5(xs: number[], ys: number[]) {
  const m = 6, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
  xs.forEach((x, i) => {
    const u = x / 1.5 - 1, p = [1, u, u * u, u ** 3, u ** 4, u ** 5];
    for (let a = 0; a < m; a++) { for (let b = 0; b < m; b++) A[a][b] += p[a] * p[b]; A[a][m] += p[a] * ys[i]; }
  });
  for (let c = 0; c < m; c++) {
    let piv = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r < m; r++) if (r !== c) { const k = A[r][c] / A[c][c]; for (let j = c; j <= m; j++) A[r][j] -= k * A[c][j]; }
  }
  const w = A.map((row, i) => row[m] / row[i]);
  return (x: number) => { const u = x / 1.5 - 1; return w[0] + u * (w[1] + u * (w[2] + u * (w[3] + u * (w[4] + u * w[5])))); };
}

/** 고정 모델 μ̂와 참 포함률 함수 cov(q) */
export function makeModel(seed = 21) {
  const r = rng(seed);
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < 500; i++) { const x = 3 * r.u(); xs.push(x); ys.push(f(x) + r.normal() * sdTrue(x)); }
  const mu = polyfit5(xs, ys);
  const grid = Array.from({ length: 2000 }, () => 3 * r.u());
  const bias = grid.map((x) => f(x) - mu(x)), sd = grid.map(sdTrue);
  const trueCoverage = (q: number) => { let s = 0; for (let i = 0; i < grid.length; i++) s += Phi((q - bias[i]) / sd[i]) - Phi((-q - bias[i]) / sd[i]); return s / grid.length; };
  const QMAX = 4, NQ = 1601, table = Float64Array.from({ length: NQ }, (_, i) => trueCoverage((QMAX * i) / (NQ - 1)));
  const cov = (q: number) => {
    if (!Number.isFinite(q) || q >= QMAX) return 1;
    const t = (q / QMAX) * (NQ - 1), i = Math.floor(t);
    return table[i] + (t - i) * (table[i + 1] - table[i]);
  };
  return { mu, cov, trueCoverage, train: { xs, ys } };
}

/** 보정 세트 REPS개의 정렬된 점수 (REPS × n) */
export function calibrationScores(model: ReturnType<typeof makeModel>, n: number) {
  const r = rng(7000 + n), out = new Float64Array(REPS * n);
  for (let k = 0; k < REPS; k++) {
    const s = out.subarray(k * n, (k + 1) * n);
    for (let i = 0; i < n; i++) { const x = 3 * r.u(); s[i] = Math.abs(f(x) + r.normal() * sdTrue(x) - model.mu(x)); }
    s.sort();
  }
  return out;
}

export const kOf = (n: number, a: number) => Math.ceil((n + 1) * (1 - a) - 1e-9);

export function coverages(model: ReturnType<typeof makeModel>, sorted: Float64Array, n: number, a: number) {
  const k = kOf(n, a), cv = new Float64Array(REPS);
  for (let j = 0; j < REPS; j++) cv[j] = k > n ? 1 : model.cov(sorted[j * n + k - 1]);
  return { k, cv };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const fr = frame(root, "split conformal 포함률", "simulate", narrow ? 0.7 : 0.5, "보정 세트 1000개로 만든 구간의 참 포함률 히스토그램과 이론 분포 Beta(k, n+1−k) 곡선");
  const model = makeModel();
  let ni = 3, a = 0.1; // n = 19 (장의 예)
  let sorted = calibrationScores(model, NS[ni]);
  let res = coverages(model, sorted, NS[ni], a);

  function draw() {
    const { ctx, width: W, height: H } = fr;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const n = NS[ni], { k, cv } = res;
    const PL = 14, PR = 14, top = 40, bottom = H - 24;
    const thr = 1 - a - 0.05;
    if (k > n) {
      ctx.fillStyle = C.edge; ctx.font = "600 14px Pretendard Variable, sans-serif";
      const t1 = `k = ${k} > n = ${n}: q̂ = ∞`, t2 = "구간이 무한해 포함률은 언제나 1이다";
      ctx.fillText(t1, (W - ctx.measureText(t1).width) / 2, H / 2 - 8);
      ctx.fillStyle = C.mid; ctx.font = "13px Pretendard Variable, sans-serif";
      ctx.fillText(t2, (W - ctx.measureText(t2).width) / 2, H / 2 + 14);
      return;
    }
    const b = n + 1 - k, mean = k / (n + 1), sd = Math.sqrt((k * b) / ((n + 1) ** 2 * (n + 2)));
    const x0 = Math.max(0, Math.min(thr - 0.02, mean - 4.5 * sd)), x1 = 1;
    const X = (v: number) => PL + ((v - x0) / (x1 - x0)) * (W - PL - PR);
    const NB = 40, bw = (x1 - x0) / NB, hist = new Array(NB).fill(0);
    for (const v of cv) { const i = Math.min(NB - 1, Math.max(0, Math.floor((v - x0) / bw))); hist[i]++; }
    const dens = hist.map((h) => h / (REPS * bw));
    let pmax = 0; for (let i = 0; i <= 200; i++) pmax = Math.max(pmax, betaPdf(k, b, x0 + ((x1 - x0) * i) / 200));
    const ymax = Math.max(pmax, ...dens) * 1.1;
    const Y = (v: number) => bottom - (v / ymax) * (bottom - top);
    // 막대: 기준(1 − α − 0.05) 미만은 코랄
    for (let i = 0; i < NB; i++) {
      const lo = x0 + i * bw;
      ctx.fillStyle = lo + bw <= thr + 1e-12 ? alpha(C.edge, 0.75) : alpha(C.ink2, 0.45);
      ctx.fillRect(X(lo) + 0.5, Y(dens[i]), Math.max(1, X(lo + bw) - X(lo) - 1), bottom - Y(dens[i]));
    }
    // 이론 Beta 밀도
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 240; i++) { const v = x0 + ((x1 - x0) * i) / 240, y = Y(betaPdf(k, b, v)); if (i) ctx.lineTo(X(v), y); else ctx.moveTo(X(v), y); }
    ctx.stroke();
    // 축과 눈금
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, bottom); ctx.lineTo(W - PR, bottom); ctx.stroke();
    ctx.fillStyle = C.mid;
    const span = x1 - x0, tick = span > 0.5 ? 0.1 : span > 0.2 ? 0.05 : span > 0.08 ? 0.02 : 0.01;
    let last = -Infinity;
    for (let v = Math.ceil(x0 / tick - 1e-9) * tick; v <= x1 + 1e-9; v += tick) {
      const s = v.toFixed(2), w = ctx.measureText(s).width, x = Math.min(W - PR - w, Math.max(PL, X(v) - w / 2));
      if (x < last + 6) continue;
      ctx.fillText(s, x, bottom + 15); last = x + w;
    }
    // 목표 1 − α
    ctx.strokeStyle = C.good; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(1 - a), top - 4); ctx.lineTo(X(1 - a), bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.good; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.ink; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(`이론 Beta(${k}, ${b}): 평균 ${mean.toFixed(3)} · sd ${sd.toFixed(3)}`, PL, 16);
    ctx.fillStyle = C.good; ctx.font = "600 12px Pretendard Variable, sans-serif";
    const tl = `1 − α = ${(1 - a).toFixed(2)}`;
    ctx.fillText(tl, Math.max(PL, Math.min(W - PR - ctx.measureText(tl).width, X(1 - a) - ctx.measureText(tl).width - 6)), top - 8);
  }

  let raf = 0;
  function update(resample: boolean) {
    const n = NS[ni];
    if (resample) sorted = calibrationScores(model, n);
    res = coverages(model, sorted, n, a);
    const { k, cv } = res;
    const thr = 1 - a - 0.05;
    if (k > n) {
      ro.set("k", `k = ${k} > n = ${n}: 필요한 n ≥ ${Math.ceil(1 / a - 1 - 1e-9)}`, "hot");
      ro.set("mean", "포함률 1(구간 무한)"); ro.set("low", "—");
    } else {
      const b = n + 1 - k, m = cv.reduce((s, v) => s + v, 0) / REPS;
      const sd = Math.sqrt(cv.reduce((s, v) => s + (v - m) ** 2, 0) / REPS);
      const low = cv.reduce((s, v) => s + (v < thr ? 1 : 0), 0) / REPS;
      ro.set("k", `k = ${k}, n = ${n}`);
      ro.set("mean", `평균 ${m.toFixed(3)} · sd ${sd.toFixed(3)}`, "ok");
      ro.set("low", `${thr.toFixed(2)} 미만 ${(100 * low).toFixed(1)}% (이론 ${(100 * betaI(k, b, thr)).toFixed(1)}%)`, low > 0.05 ? "hot" : "");
    }
    pageState.set("conformal", { n, alpha: a, k }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pn = param({ label: "n", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => String(NS[v]), aria: "보정 세트 크기 n", onInput: (v) => { ni = v; pn.input.setAttribute("aria-valuetext", String(NS[v])); update(true); } });
  pn.input.setAttribute("aria-valuetext", String(NS[ni]));
  const pa = param({ label: "α", min: 0.01, max: 0.3, step: 0.01, value: a, fmt: (v) => v.toFixed(2), aria: "미포함 허용률 α", onInput: (v) => { a = v; update(false); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row, pa.row);
  const ro = readouts();
  fr.panel.append(grid, ro.row);
  fr.onResize(draw);
  update(false);
}
