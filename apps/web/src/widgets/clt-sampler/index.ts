// 중심극한정리와 꼬리 (data/widgets.toml: clt-sampler).
// 분포에서 크기 n인 표본을 4000번 뽑아 T = (x̄ − μ)/(s/√n)(s는 ddof = 1)의 히스토그램을 N(0, 1)과 겹쳐 그린다.
// |T| ≤ 1.96이 곧 근사 95% 구간 x̄ ± 1.96·s/√n이 참 평균 μ를 포함하는 사건이므로, 포함률 = 가운데 막대의 비율이다(11.5장 실습과 같은 기준).
// Cauchy는 평균이 없어 μ 대신 중심 0을 쓰고 포함률을 내지 않는다. 표본 평균의 사분위 범위(IQR)는 모든 분포에서 함께 보인다
// (분산이 유한하면 1/√n로 줄고, Cauchy에서는 n과 상관없이 약 2다).
import { frame, param, presets, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

type Dist = "unif" | "exp" | "par25" | "par15" | "cauchy";
export const DISTS: Record<Dist, { label: string; mu: number | null; finiteVar: boolean; draw: (u: number) => number }> = {
  unif: { label: "균등 U(0,1)", mu: 0.5, finiteVar: true, draw: (u) => u },
  exp: { label: "지수 Exp(1)", mu: 1, finiteVar: true, draw: (u) => -Math.log(1 - u) },
  par25: { label: "Pareto α=2.5", mu: 2.5 / 1.5, finiteVar: true, draw: (u) => Math.pow(1 - u, -1 / 2.5) },
  par15: { label: "Pareto α=1.5", mu: 3, finiteVar: false, draw: (u) => Math.pow(1 - u, -1 / 1.5) },
  cauchy: { label: "Cauchy", mu: null, finiteVar: false, draw: (u) => Math.tan(Math.PI * (u - 0.5)) },
};
export const NS = [2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];
export const REPS = 4000;
const TB = 4.9, NB = 40; // 막대 폭 0.245 = 1.96/8: ±1.96이 막대 경계와 맞는다

/** 반복을 프레임마다 나눠 돌리는 표본 추출 작업. step(draws)는 표본을 대략 draws개 더 뽑고, 다 돌았으면 true를 낸다. */
export function job(dist: Dist, n: number, seed: number, reps = REPS) {
  const D = DISTS[dist], r = rng(seed), mu = D.mu ?? 0;
  const T = new Float64Array(reps), means = new Float64Array(reps), xs = new Float64Array(n);
  let k = 0;
  return {
    get count() { return k; },
    step(draws: number) {
      const end = Math.min(reps, k + Math.max(1, Math.ceil(draws / n)));
      for (; k < end; k++) {
        let s = 0, ss = 0;
        for (let i = 0; i < n; i++) { xs[i] = D.draw(r.u()); s += xs[i]; }
        const m = s / n;
        // 분산은 평균을 먼저 구한 뒤 편차 제곱을 더한다(큰 값이 나오는 Pareto·Cauchy에서 상쇄 오차를 줄인다)
        for (let i = 0; i < n; i++) ss += (xs[i] - m) ** 2;
        means[k] = m;
        T[k] = (m - mu) / (Math.sqrt(ss / (n - 1)) / Math.sqrt(n));
      }
      return k >= reps;
    },
    /** 지금까지 뽑은 k번의 요약 */
    result() {
      let lo = 0, hi = 0, out = 0;
      const hist = new Array(NB).fill(0);
      for (let j = 0; j < k; j++) {
        const t = T[j];
        if (t < -1.96) lo++; else if (t > 1.96) hi++;
        const b = Math.floor(((t + TB) / (2 * TB)) * NB);
        if (b >= 0 && b < NB) hist[b]++; else out++;
      }
      const sorted = means.slice(0, k).sort();
      const q = (p: number) => { const x = p * (k - 1), i = Math.floor(x); return sorted[i] + (x - i) * ((sorted[i + 1] ?? sorted[i]) - sorted[i]); };
      const K = Math.max(k, 1);
      return { reps: k, hist, out: out / K, lo: lo / K, hi: hi / K, cover: 1 - (lo + hi) / K, iqr: k > 1 ? q(0.75) - q(0.25) : 0 };
    },
  };
}
export function simulate(dist: Dist, n: number, seed: number, reps = REPS) {
  const j = job(dist, n, seed, reps);
  j.step(Infinity);
  return j.result();
}

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "중심극한정리와 꼬리", "manipulate", narrow ? 0.72 : 0.5, "표준화한 표본 평균 T의 히스토그램과 표준정규분포 곡선, ±1.96 경계");
  let dist: Dist = "exp", ni = NS.indexOf(30), seed = 1;
  let work = job(dist, NS[ni], seed);
  let res = work.result();

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 34, R = 12, T = 12, B = 24;
    const X = (t: number) => L + ((t + TB) / (2 * TB)) * (W - L - R);
    const dens = res.hist.map((c) => c / Math.max(res.reps, 1) / ((2 * TB) / NB));
    const ymax = Math.max(0.5, ...dens) * 1.08;
    const Y = (d: number) => T + (1 - d / ymax) * (H - T - B);
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
    ctx.textAlign = "center";
    for (let k = -4; k <= 4; k++) if (k % (narrow ? 2 : 1) === 0 || k === 0) ctx.fillText(String(k).replace("-", "−"), X(k), H - 8);
    ctx.textAlign = "right";
    for (const v of [0.2, 0.4]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillText(String(v), L - 5, Y(v) + 4); }
    ctx.textAlign = "start";
    // 막대: |T| ≤ 1.96 안쪽은 파랑, 바깥은 코랄
    const bw = (W - L - R) / NB;
    res.hist.forEach((_, b) => {
      const t0 = -TB + (b * 2 * TB) / NB, t1 = t0 + (2 * TB) / NB;
      const inside = t0 >= -1.96 - 1e-9 && t1 <= 1.96 + 1e-9;
      ctx.fillStyle = inside ? C.ink2 : C.edge;
      ctx.globalAlpha = inside ? 0.55 : 0.85;
      ctx.fillRect(X(t0) + 0.5, Y(dens[b]), bw - 1, Y(0) - Y(dens[b]));
    });
    ctx.globalAlpha = 1;
    // ±1.96
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    for (const z of [-1.96, 1.96]) { ctx.beginPath(); ctx.moveTo(X(z), T); ctx.lineTo(X(z), Y(0)); ctx.stroke(); }
    ctx.setLineDash([]);
    // N(0, 1)
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const t = -TB + (i / 200) * 2 * TB, d = Math.exp(-0.5 * t * t) / Math.sqrt(2 * Math.PI); i ? ctx.lineTo(X(t), Y(d)) : ctx.moveTo(X(t), Y(d)); }
    ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "12px Pretendard Variable, sans-serif";
    ctx.fillText(DISTS[dist].mu === null ? "T = x̄/(s/√n)" : "T = (x̄ − μ)/(s/√n)", L + 6, T + 12);
    ctx.fillStyle = C.ink;
    ctx.fillText("— N(0, 1)", L + 6, T + 28);
    if (res.out > 0) { ctx.fillStyle = C.mid; ctx.textAlign = "right"; ctx.fillText(`|T| > 4.9: ${(res.out * 100).toFixed(1)}%`, W - R - 4, T + 12); ctx.textAlign = "start"; }
  }

  function report() {
    const D = DISTS[dist];
    if (D.mu === null) ro.set("cov", "포함률: 평균이 없다", "hot");
    else ro.set("cov", `포함률 ${res.cover.toFixed(3)}`, res.cover < 0.93 ? "hot" : "ok");
    ro.set("tail", `P(T < −1.96) ${res.lo.toFixed(3)}`);
    ro.set("tail2", `P(T > 1.96) ${res.hi.toFixed(3)}`);
    ro.set("iqr", `표본 평균의 IQR ${res.iqr < 0.01 ? res.iqr.toExponential(1) : res.iqr.toFixed(3)}`, D.mu === null ? "hot" : "");
    ro.set("n", `n = ${NS[ni]} · 반복 ${res.reps}번${D.finiteVar ? "" : " · 분산 무한"}`);
    draw();
  }
  // 큰 n에서도 조작이 끊기지 않도록 한 프레임에 표본 약 25만 개씩 뽑고, 그때까지의 결과를 그린다.
  let raf = 0;
  function update() {
    cancelAnimationFrame(raf);
    work = job(dist, NS[ni], seed);
    pageState.set("clt", { dist, n: NS[ni] }, root);
    const tick = () => {
      const done = work.step(250_000);
      res = work.result();
      report();
      if (!done) raf = requestAnimationFrame(tick);
    };
    tick();
  }

  const SHORT: Record<Dist, string> = { unif: "균등", exp: "지수", par25: "Pareto 2.5", par15: "Pareto 1.5", cauchy: "Cauchy" };
  const pre = presets((Object.keys(DISTS) as Dist[]).map((k) => ({ key: k, label: narrow ? SHORT[k] : DISTS[k].label })), (k) => { dist = k; update(); }, dist);
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 8px"));
  const pn = param({ label: "n", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => String(NS[v]), aria: "표본 크기 n", onInput: (v) => { ni = v; update(); } });
  pn.input.setAttribute("aria-valuetext", String(NS[ni]));
  pn.input.addEventListener("input", () => pn.input.setAttribute("aria-valuetext", String(NS[ni])));
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const tools = el("div", "wframe-row");
  const again = el("button", "preset");
  again.type = "button"; again.textContent = "다시 뽑기";
  again.addEventListener("click", () => { seed += 1; update(); });
  tools.append(again);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
