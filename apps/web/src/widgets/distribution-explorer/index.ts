// 분포 탐색기 (data/widgets.toml: distribution-explorer).
// 위: 확률밀도(pdf) 또는 확률질량(pmf), 아래: 누적분포(cdf). x 아래의 넓이(또는 막대 합)가 곧 F(x)임을 본다.
// 조작: 분포 프리셋, 모수 슬라이더, x 슬라이더(그림을 끌어도 된다). 평균·분산은 닫힌 식으로, cdf는 정규화 불완전 감마·베타 함수로 계산한다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

// ── 특수함수 ────────────────────────────────────────────────
const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
/** log Γ(x) (Lanczos 근사, x > 0) */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
/** 정규화 하측 불완전 감마 P(a, x) */
export function gammaP(a: number, x: number): number {
  if (x <= 0) return 0;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 500; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
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
/** 정규화 불완전 베타 I_x(a, b) */
export function betaI(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
const normCdf = (z: number) => 0.5 * (1 + Math.sign(z) * gammaP(0.5, (z * z) / 2));

// ── 분포 ────────────────────────────────────────────────────
type P = number[];
interface Spec { sym: string; aria: string; min: number; max: number; step: number; value: number; fmt?: (v: number) => string }
export interface Dist {
  label: string; name: (p: P) => string; discrete: boolean; params: Spec[];
  range: (p: P) => [number, number]; cap: number;
  pdf: (x: number, p: P) => number; cdf: (x: number, p: P) => number; mean: (p: P) => number; variance: (p: P) => number;
}
const f2 = (v: number) => v.toFixed(2);
const int = (v: number) => String(Math.round(v));
const binLog = (n: number, k: number) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
const sumTo = (x: number, p: P, pmf: (k: number, p: P) => number, hi = Infinity) => { let s = 0; for (let k = 0; k <= Math.min(Math.floor(x), hi); k++) s += pmf(k, p); return Math.min(1, s); };
const poisPmf = (k: number, [l]: P) => (k < 0 || k !== Math.floor(k) ? 0 : Math.exp(k * Math.log(l) - l - lgamma(k + 1)));
const binPmf = (k: number, [n, q]: P) => (k < 0 || k > n || k !== Math.floor(k) ? 0 : Math.exp(binLog(n, k) + k * Math.log(q) + (n - k) * Math.log(1 - q)));

export const DISTS = {
  normal: {
    label: "정규", name: ([m, s]) => `N(${f2(m)}, ${f2(s)}²)`, discrete: false, cap: Infinity,
    params: [{ sym: "μ", aria: "평균 μ", min: -3, max: 3, step: 0.1, value: 0 }, { sym: "σ", aria: "표준편차 σ", min: 0.3, max: 2.5, step: 0.05, value: 1 }],
    range: () => [-7, 7],
    pdf: (x, [m, s]) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI)),
    cdf: (x, [m, s]) => normCdf((x - m) / s), mean: ([m]) => m, variance: ([, s]) => s * s,
  },
  exponential: {
    label: "지수", name: ([l]) => `Exp(λ=${f2(l)})`, discrete: false, cap: Infinity,
    params: [{ sym: "λ", aria: "비율 λ", min: 0.25, max: 4, step: 0.05, value: 1 }],
    range: () => [0, 8],
    pdf: (x, [l]) => (x < 0 ? 0 : l * Math.exp(-l * x)), cdf: (x, [l]) => (x < 0 ? 0 : 1 - Math.exp(-l * x)),
    mean: ([l]) => 1 / l, variance: ([l]) => 1 / (l * l),
  },
  gamma: {
    label: "감마", name: ([a, b]) => `Gamma(α=${f2(a)}, β=${f2(b)})`, discrete: false, cap: 2,
    params: [{ sym: "α", aria: "모양 α", min: 0.5, max: 8, step: 0.1, value: 2 }, { sym: "β", aria: "비율 β", min: 0.5, max: 4, step: 0.05, value: 1 }],
    range: () => [0, 16],
    pdf: (x, [a, b]) => (x <= 0 ? 0 : Math.exp(a * Math.log(b) + (a - 1) * Math.log(x) - b * x - lgamma(a))),
    cdf: (x, [a, b]) => gammaP(a, b * x), mean: ([a, b]) => a / b, variance: ([a, b]) => a / (b * b),
  },
  beta: {
    label: "베타", name: ([a, b]) => `Beta(${f2(a)}, ${f2(b)})`, discrete: false, cap: 6,
    params: [{ sym: "α", aria: "모수 α", min: 0.5, max: 8, step: 0.1, value: 2 }, { sym: "β", aria: "모수 β", min: 0.5, max: 8, step: 0.1, value: 5 }],
    range: () => [0, 1],
    pdf: (x, [a, b]) => (x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + lgamma(a + b) - lgamma(a) - lgamma(b))),
    cdf: (x, [a, b]) => betaI(a, b, x), mean: ([a, b]) => a / (a + b), variance: ([a, b]) => (a * b) / ((a + b) ** 2 * (a + b + 1)),
  },
  poisson: {
    label: "포아송", name: ([l]) => `Poisson(λ=${f2(l)})`, discrete: true, cap: Infinity,
    params: [{ sym: "λ", aria: "평균 발생 수 λ", min: 0.5, max: 15, step: 0.1, value: 3 }],
    range: () => [-0.5, 25.5],
    pdf: poisPmf, cdf: (x, p) => sumTo(x, p, poisPmf), mean: ([l]) => l, variance: ([l]) => l,
  },
  binomial: {
    label: "이항", name: ([n, q]) => `Bin(n=${n}, p=${f2(q)})`, discrete: true, cap: Infinity,
    params: [{ sym: "n", aria: "시행 수 n", min: 1, max: 30, step: 1, value: 10, fmt: int }, { sym: "p", aria: "성공 확률 p", min: 0.05, max: 0.95, step: 0.05, value: 0.3 }],
    range: ([n]) => [-0.5, Math.max(10, n) + 0.5],
    pdf: binPmf, cdf: (x, p) => sumTo(x, p, binPmf, p[0]), mean: ([n, q]) => n * q, variance: ([n, q]) => n * q * (1 - q),
  },
} satisfies Record<string, Dist>;
type Key = keyof typeof DISTS;

/** 1·2·2.5·5 × 10^k 단위로 올림한 축 최댓값 */
function nice(v: number) {
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * e >= v) return m * e;
  return 10 * e;
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "분포 탐색기", "manipulate", 0.66, "선택한 분포의 확률밀도 또는 확률질량과 누적분포 곡선, x 이하의 넓이");
  let key: Key = (attrs.dist in DISTS ? attrs.dist : "normal") as Key;
  let p: P = [];
  let x = 0;
  const D = () => DISTS[key] as Dist;

  const PAD_L = 40, PAD_R = 14;
  const layout = () => {
    const H = f.height, top = 18, split = Math.round(H * 0.62), bottom = H - 22;
    return { top, pdfBase: split - 22, cdfTop: split + 4, cdfBase: bottom };
  };
  const X = (v: number) => { const [lo, hi] = D().range(p); return PAD_L + ((v - lo) / (hi - lo)) * (f.width - PAD_L - PAD_R); };
  const invX = (px: number) => { const [lo, hi] = D().range(p); return lo + ((px - PAD_L) / (f.width - PAD_L - PAD_R)) * (hi - lo); };

  function draw() {
    const { ctx, width: W } = f;
    const d = D();
    const [lo, hi] = d.range(p);
    const L = layout();
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, f.height);
    ctx.font = "11px JetBrains Mono, monospace";

    // pdf · pmf 값과 축 최댓값
    const N = 400;
    const grid = d.discrete
      ? Array.from({ length: Math.round(hi - lo) }, (_, i) => Math.ceil(lo) + i)
      : Array.from({ length: N }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / N);
    const vals = grid.map((g) => d.pdf(g, p));
    const finite = vals.filter((v) => Number.isFinite(v));
    const ymax = nice(Math.max(1e-6, Math.min(Math.max(...finite) * 1.08, d.cap)));
    const Y = (v: number) => L.pdfBase - (Math.min(v, ymax) / ymax) * (L.pdfBase - L.top);
    const Yc = (v: number) => L.cdfBase - v * (L.cdfBase - L.cdfTop);

    // 격자와 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (const t of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(PAD_L, Y(t * ymax)); ctx.lineTo(W - PAD_R, Y(t * ymax)); ctx.stroke();
      ctx.fillText(String(parseFloat((t * ymax).toPrecision(3))), 4, Y(t * ymax) + 4);
      ctx.beginPath(); ctx.moveTo(PAD_L, Yc(t)); ctx.lineTo(W - PAD_R, Yc(t)); ctx.stroke();
      ctx.fillText(t.toFixed(1), 4, Yc(t) + 4);
    }
    const step = tickStep(hi - lo, W);
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
      const s = fmtTick(t);
      ctx.fillText(s, X(t) - ctx.measureText(s).width / 2, L.pdfBase + 14);
    }

    const edge = C.edge;
    ctx.save();
    ctx.beginPath(); ctx.rect(PAD_L, L.top - 4, W - PAD_L - PAD_R, L.pdfBase - L.top + 4); ctx.clip();
    if (d.discrete) {
      const bw = Math.max(2, Math.min(18, ((W - PAD_L - PAD_R) / (hi - lo)) * 0.6));
      grid.forEach((k, i) => {
        ctx.fillStyle = k <= x ? edge : alpha(C.mid, 0.45);
        ctx.fillRect(X(k) - bw / 2, Y(vals[i]), bw, L.pdfBase - Y(vals[i]));
      });
    } else {
      // x 이하의 넓이 = F(x)
      ctx.fillStyle = alpha(edge, 0.22);
      ctx.beginPath(); ctx.moveTo(X(lo), L.pdfBase);
      grid.forEach((g, i) => { if (g <= x) ctx.lineTo(X(g), Y(vals[i])); });
      ctx.lineTo(X(Math.min(x, hi)), Y(d.pdf(Math.min(x, hi), p))); ctx.lineTo(X(Math.min(x, hi)), L.pdfBase); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
      ctx.beginPath(); grid.forEach((g, i) => (i ? ctx.lineTo(X(g), Y(vals[i])) : ctx.moveTo(X(g), Y(vals[i])))); ctx.stroke();
    }
    ctx.restore();

    // 평균 ± 표준편차
    const m = d.mean(p), sd = Math.sqrt(d.variance(p));
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(m), L.top); ctx.lineTo(X(m), L.pdfBase); ctx.stroke(); ctx.setLineDash([]);
    ctx.lineWidth = 3; ctx.strokeStyle = C.ink2;
    const yb = L.pdfBase - 4;
    ctx.beginPath(); ctx.moveTo(X(Math.max(lo, m - sd)), yb); ctx.lineTo(X(Math.min(hi, m + sd)), yb); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    const ml = "E[X]";
    ctx.fillText(ml, Math.min(X(m) + 5, W - PAD_R - ctx.measureText(ml).width), L.top + 8);
    ctx.fillText(d.discrete ? "pmf" : "pdf", PAD_L + 4, L.top + 8);

    // cdf
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    if (d.discrete) {
      let prev = 0;
      ctx.moveTo(X(lo), Yc(0));
      for (const k of grid) { ctx.lineTo(X(k), Yc(prev)); prev = d.cdf(k, p); ctx.lineTo(X(k), Yc(prev)); }
      ctx.lineTo(X(hi), Yc(prev));
    } else {
      for (let i = 0; i <= N; i++) { const g = lo + ((hi - lo) * i) / N; const yy = Yc(d.cdf(g, p)); i ? ctx.lineTo(X(g), yy) : ctx.moveTo(X(g), yy); }
    }
    ctx.stroke();
    ctx.fillStyle = C.mid; ctx.fillText("cdf", PAD_L + 4, L.cdfTop + 10);

    // 현재 x
    const F = d.cdf(x, p);
    ctx.strokeStyle = edge; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(X(x), L.top); ctx.lineTo(X(x), L.cdfBase); ctx.stroke();
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(PAD_L, Yc(F)); ctx.lineTo(X(x), Yc(F)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = edge;
    ctx.beginPath(); ctx.arc(X(x), Yc(F), 5, 0, 7); ctx.fill();
    ctx.font = "600 12px JetBrains Mono, monospace";
    const lab = `F(x) = ${F.toFixed(3)}`;
    const lw = ctx.measureText(lab).width;
    ctx.fillText(lab, X(x) + 10 + lw > W - PAD_R ? X(x) - 10 - lw : X(x) + 10, Math.max(L.cdfTop + 12, Yc(F) - 8));
  }
  const tickStep = (span: number, W: number) => { const target = span / Math.max(3, Math.floor((W - 60) / 56)); return nice(target); };
  const fmtTick = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : Math.abs(v) < 1 ? v.toFixed(v * 10 === Math.round(v * 10) ? 1 : 2) : v.toFixed(1));

  // 조작
  const grid = el("div", "param-grid");
  let px: ReturnType<typeof param>;
  function build() {
    const d = D();
    p = d.params.map((s) => s.value);
    grid.innerHTML = "";
    d.params.forEach((s, i) => {
      const r = param({ label: s.sym, min: s.min, max: s.max, step: s.step, value: s.value, fmt: s.fmt ?? f2, aria: s.aria, onInput: (v) => { p[i] = v; clampX(); update(); } });
      grid.append(r.row);
    });
    const [lo, hi] = d.range(p);
    x = d.discrete ? Math.round(d.mean(p)) : Math.round(d.mean(p) * 10) / 10;
    const xr = d.discrete ? [0, key === "binomial" ? 30 : Math.floor(hi)] : [lo, hi];
    px = param({ label: "x", min: xr[0], max: xr[1], step: d.discrete ? 1 : (hi - lo) / 200, value: x, fmt: d.discrete ? int : f2, aria: "누적확률을 볼 위치 x", onInput: (v) => { x = v; update(); } });
    grid.append(px.row);
  }
  const clampX = () => {
    const [lo, hi] = D().range(p);
    const nx = D().discrete ? Math.max(0, Math.min(x, Math.floor(hi))) : Math.max(lo, Math.min(x, hi));
    if (nx !== x) { x = nx; px.set(x); }
  };
  const pre = presets((Object.keys(DISTS) as Key[]).map((k) => ({ key: k, label: DISTS[k].label })), (k) => { key = k; build(); update(); }, key);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);

  function update() {
    const d = D();
    ro.set("name", d.name(p));
    ro.set("mean", `E[X] = ${d.mean(p).toFixed(3)}`);
    ro.set("var", `Var[X] = ${d.variance(p).toFixed(3)}`);
    ro.set("cdf", `P(X ≤ ${d.discrete ? Math.floor(x) : x.toFixed(2)}) = ${d.cdf(x, p).toFixed(3)}`, "hot");
    pageState.set("dist", { name: key, params: [...p] }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  let dragging = false;
  const setX = (e: PointerEvent) => {
    const r = f.canvas.getBoundingClientRect();
    const [lo, hi] = D().range(p);
    let v = Math.max(lo, Math.min(hi, invX(e.clientX - r.left)));
    v = D().discrete ? Math.max(0, Math.round(v)) : Math.round(v * 100) / 100;
    x = v; px.set(v); update();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); setX(e); });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) setX(e); });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.canvas.style.cursor = "ew-resize";
  f.onResize(draw);
  build();
  update();
  draw();
}
