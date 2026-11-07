// 순차적 베이지안 갱신 (data/widgets.toml: bayesian-update).
// 동전의 앞면 확률 θ에 Beta(α₀, β₀) 사전분포를 두고, 던진 결과를 하나씩 더한다. 켤레성 때문에 사후분포는
//   Beta(α₀ + 앞면 수, β₀ + 뒷면 수)
// 이고, 한 번 갱신한 사후가 다음 관측의 사전이 된다. 사전 평균 m과 강도 κ = α₀ + β₀로 사전을 정한다(α₀ = mκ, β₀ = (1 − m)κ).
// 동전 결과는 참 θ* = 0.7에서 고정 시드로 뽑으므로 관측 수 n을 되돌리면 같은 순서가 다시 나온다.
import { frame, param, presets, readouts, el, C, rng, alpha, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

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
/** 정규화 불완전 베타 I_x(a, b) = Beta(a, b)의 cdf */
function betaI(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
/** Beta(a, b)의 q 분위수(이분법) */
export function betaQuantile(a: number, b: number, q: number) {
  let lo = 0, hi = 1;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (betaI(a, b, mid) < q) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
const betaPdf = (x: number, a: number, b: number) =>
  x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + lgamma(a + b) - lgamma(a) - lgamma(b));

export const THETA = 0.7, NMAX = 100;
/** 고정 시드로 뽑은 동전 결과(1 = 앞면) */
export const TOSSES = (() => { const r = rng(4242); return Array.from({ length: NMAX }, () => (r.u() < THETA ? 1 : 0)); })();
export function posterior(m: number, kappa: number, n: number) {
  const h = TOSSES.slice(0, n).reduce<number>((s, v) => s + v, 0);
  const a0 = m * kappa, b0 = (1 - m) * kappa;
  return { h, a0, b0, a: a0 + h, b: b0 + n - h };
}
const nice = (v: number) => { const e = Math.pow(10, Math.floor(Math.log10(v))); for (const k of [1, 2, 2.5, 5, 10]) if (k * e >= v) return k * e; return 10 * e; };

const PRIORS = {
  flat: { label: "균등", m: 0.5, lk: 1 },
  weak: { label: "약한 믿음", m: 0.5, lk: 3 },
  wrong: { label: "강한 편향", m: 0.25, lk: 6 },
} as const;
type PK = keyof typeof PRIORS;

export function mount(root: HTMLElement) {
  const f = frame(root, "순차적 베이지안 갱신", "stepper", 0.62, "동전 앞면 확률의 사전분포, 우도, 사후분포와 95% 신용구간, 지금까지의 관측");
  let m: number = PRIORS.flat.m, lk: number = PRIORS.flat.lk, n = 0; // κ = 2^lk
  const kappa = () => Math.pow(2, lk);
  let shown = { a: 1, b: 1 }; // 그려진 사후(갱신하면 새 값으로 미끄러진다)
  let anim = 0;

  const PAD_L = 36, PAD_R = 14;
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const post = posterior(m, kappa(), n);
    const cols = W < 520 ? 25 : 50, rows = NMAX / cols, cell = Math.min(10, (W - PAD_L - PAD_R) / cols);
    const stripH = rows * (cell + 4);
    const top = 30, base = H - 24 - stripH - 10;
    const X = (t: number) => PAD_L + t * (W - PAD_L - PAD_R);
    const N = 300;
    const ts = Array.from({ length: N + 1 }, (_, i) => Math.min(1 - 1e-6, Math.max(1e-6, i / N)));
    const prior = ts.map((t) => betaPdf(t, post.a0, post.b0));
    const like = n ? ts.map((t) => betaPdf(t, post.h + 1, n - post.h + 1)) : null; // 넓이 1로 정규화한 우도
    const pst = ts.map((t) => betaPdf(t, shown.a, shown.b));
    const inner = (arr: number[]) => Math.max(...arr.slice(2, -2).filter(Number.isFinite));
    const ymax = nice(Math.max(1.2, inner(pst), inner(prior), like ? inner(like) : 0) * 1.08);
    const Y = (v: number) => base - (Math.min(v, ymax * 1.05) / ymax) * (base - top);

    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (const k of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(PAD_L, Y(k * ymax)); ctx.lineTo(W - PAD_R, Y(k * ymax)); ctx.stroke();
      ctx.fillText(String(parseFloat((k * ymax).toPrecision(3))), 4, Y(k * ymax) + 4);
    }
    for (let k = 0; k <= 10; k += 2) { const s = (k / 10).toFixed(1); ctx.fillText(s, X(k / 10) - ctx.measureText(s).width / 2, base + 14); }

    ctx.save();
    ctx.beginPath(); ctx.rect(PAD_L, top - 6, W - PAD_L - PAD_R, base - top + 6); ctx.clip();
    // 95% 신용구간
    const lo = betaQuantile(shown.a, shown.b, 0.025), hi = betaQuantile(shown.a, shown.b, 0.975);
    ctx.fillStyle = alpha(C.edge, 0.2);
    ctx.beginPath(); ctx.moveTo(X(lo), base);
    ts.forEach((t, i) => { if (t >= lo && t <= hi) ctx.lineTo(X(t), Y(pst[i])); });
    ctx.lineTo(X(hi), base); ctx.closePath(); ctx.fill();
    const path = (arr: number[]) => { ctx.beginPath(); ts.forEach((t, i) => (i ? ctx.lineTo(X(t), Y(arr[i])) : ctx.moveTo(X(t), Y(arr[i])))); ctx.stroke(); };
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.6; path(prior); ctx.setLineDash([]);
    if (like) { ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.6; path(like); }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6; path(pst);
    ctx.restore();
    // 참 θ*
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(THETA), top); ctx.lineTo(X(THETA), base); ctx.stroke();
    ctx.fillStyle = C.good; ctx.fillText("θ*", X(THETA) + 4, top - 4);
    // 범례
    ctx.textAlign = "left";
    const items: [string, string, boolean][] = [["사전", C.mid, true], ...(like ? [["우도", C.ink2, false] as [string, string, boolean]] : []), ["사후", C.edge, false]];
    let lx = PAD_L + 6;
    for (const [name, col, dash] of items) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(lx, 12); ctx.lineTo(lx + 16, 12); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(name, lx + 20, 16);
      lx += 26 + ctx.measureText(name).width + 10;
    }
    // 관측 띠: 칠한 칸 = 앞면, 빈 칸 = 뒷면, 마지막 관측은 강조
    const sy = H - 10 - stripH;
    for (let i = 0; i < NMAX; i++) {
      const cx = PAD_L + (i % cols) * cell, cy = sy + Math.floor(i / cols) * (cell + 4);
      const s = cell - 2;
      if (i >= n) { ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(cx + 0.5, cy + 0.5, s, s); continue; }
      const last = i === n - 1;
      if (TOSSES[i]) { ctx.fillStyle = last ? C.edge : C.ink; ctx.fillRect(cx, cy, s + 1, s + 1); }
      else { ctx.strokeStyle = last ? C.edge : C.ink; ctx.lineWidth = last ? 2 : 1.2; ctx.strokeRect(cx + 1, cy + 1, s - 1, s - 1); }
    }
  }

  function update(slide = false) {
    const post = posterior(m, kappa(), n);
    const lo = betaQuantile(post.a, post.b, 0.025), hi = betaQuantile(post.a, post.b, 0.975);
    ro.set("n", `관측 ${n}번 · 앞면 ${post.h}`);
    ro.set("post", `사후 Beta(${post.a.toFixed(1)}, ${post.b.toFixed(1)})`);
    ro.set("mean", `사후 평균 ${(post.a / (post.a + post.b)).toFixed(3)}`, "hot");
    ro.set("ci", `95% 신용구간 [${lo.toFixed(3)}, ${hi.toFixed(3)}]`, lo <= THETA && THETA <= hi ? "ok" : "");
    ro.set("mle", n ? `MLE ${(post.h / n).toFixed(3)}` : "MLE 없음");
    pageState.set("posterior", { alpha: post.a, beta: post.b }, root);
    cancelAnimationFrame(anim);
    const target = { a: post.a, b: post.b };
    if (!slide || reducedMotion()) { shown = target; anim = requestAnimationFrame(draw); return; }
    const from = { ...shown }, t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / 320), e = 1 - Math.pow(1 - k, 3);
      shown = { a: from.a + (target.a - from.a) * e, b: from.b + (target.b - from.b) * e };
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  const pre = presets((Object.keys(PRIORS) as PK[]).map((k) => ({ key: k, label: PRIORS[k].label })), (k) => {
    m = PRIORS[k].m; lk = PRIORS[k].lk; pm.set(m); pk.set(lk); update(true);
  }, "flat");
  const pm = param({ label: "m", min: 0.05, max: 0.95, step: 0.05, value: m, aria: "사전 평균 m", onInput: (v) => { m = v; pre.select(null); update(); } });
  const pk = param({ label: "κ", min: -1, max: 7, step: 0.25, value: lk, fmt: (v) => { const k = Math.pow(2, v); return k < 10 ? k.toFixed(1) : k.toFixed(0); }, aria: "사전 강도 κ = α₀ + β₀ (로그 눈금)", onInput: (v) => { lk = v; pre.select(null); update(); } });
  const pn = param({ label: "n", min: 0, max: NMAX, step: 1, value: n, fmt: (v) => String(v), aria: "관측 수 n", onInput: (v) => { n = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pm.row, pk.row, pn.row);
  const tools = el("div", "wframe-row");
  const add = (k: number, label: string) => {
    const b = el("button", "preset");
    b.type = "button"; b.textContent = label;
    b.addEventListener("click", () => { n = Math.min(NMAX, n + k); pn.set(n); update(true); });
    tools.append(b);
  };
  add(1, "+1 던지기");
  add(10, "+10 던지기");
  const back = el("button", "preset");
  back.type = "button"; back.textContent = "관측 비우기";
  back.addEventListener("click", () => { n = 0; pn.set(0); update(true); });
  tools.append(back);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
