// GMM EM 단계 (data/widgets.toml: gmm-em). 13.4장의 1차원 GMM을 그대로 쓴다.
// 위: 데이터 히스토그램, 성분 곡선 π_k N(x | μ_k, σ_k²)와 혼합 밀도, 아래 띠의 점은 q(z_n)(책임도)로 섞은 색.
// 아래: 단계마다 log p(x | θ)와 ELBO(q, θ). E-step은 q를 사후분포로 바꿔 둘을 맞닿게 하고, M-step은 q를 고정한 채 θ를 옮겨 둘을 함께 올린다.
// 처음 q는 균등분포다(장의 ELBO 분해 예시와 같다). 세 덩어리 데이터에서는 초기값에 따라 다른 국소해에 멈춘다.
import { frame, presets, readouts, el, C, rng, rgbOf } from "../frame";

export type Theta = { pi: number[]; mu: number[]; sd: number[] };
type DataKey = "two" | "three";
const LOG2PI = Math.log(2 * Math.PI);
const SD_FLOOR = 0.05;

/** 장과 같은 생성 과정의 표본(난수 생성기만 다르다). */
export function makeData(key: DataKey): number[] {
  const r = rng(key === "two" ? 2024 : 1);
  if (key === "two") return Array.from({ length: 500 }, () => (r.u() < 0.3 ? 1.0 + 0.5 * r.normal() : 3.0 + 0.8 * r.normal()));
  const out: number[] = [];
  for (const m of [-3, 0, 0.9]) for (let i = 0; i < 150; i++) out.push(m + 0.6 * r.normal());
  return out;
}

/** log p(x_n, z_n = k | θ), (N, K) */
export function logJoint(x: number[], th: Theta): number[][] {
  const K = th.pi.length;
  return x.map((xn) => Array.from({ length: K }, (_, k) => Math.log(th.pi[k]) - 0.5 * LOG2PI - Math.log(th.sd[k]) - 0.5 * ((xn - th.mu[k]) / th.sd[k]) ** 2));
}
const lse = (a: number[]) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); };
export const logLik = (x: number[], th: Theta) => logJoint(x, th).reduce((s, row) => s + lse(row), 0);
/** E-step: 책임도 r_nk = p(z_n = k | x_n, θ) */
export function eStep(x: number[], th: Theta): number[][] {
  return logJoint(x, th).map((row) => { const z = lse(row); return row.map((v) => Math.exp(v - z)); });
}
/** M-step: 책임도로 가중한 MLE */
export function mStep(x: number[], q: number[][]): Theta {
  const K = q[0].length, N = x.length;
  const Nk = Array.from({ length: K }, (_, k) => q.reduce((s, r) => s + r[k], 0));
  const mu = Nk.map((nk, k) => q.reduce((s, r, n) => s + r[k] * x[n], 0) / nk);
  const sd = Nk.map((nk, k) => Math.max(SD_FLOOR, Math.sqrt(q.reduce((s, r, n) => s + r[k] * (x[n] - mu[k]) ** 2, 0) / nk)));
  return { pi: Nk.map((v) => v / N), mu, sd };
}
/** ELBO(q, θ) = Σ_n Σ_k q_nk (log p(x_n, z_n = k | θ) − log q_nk) */
export function elbo(x: number[], q: number[][], th: Theta): number {
  const lj = logJoint(x, th);
  let s = 0;
  for (let n = 0; n < x.length; n++) for (let k = 0; k < q[n].length; k++) if (q[n][k] > 0) s += q[n][k] * (lj[n][k] - Math.log(q[n][k]));
  return s;
}

const DATA: Record<DataKey, { label: string; K: number; range: [number, number]; inits: { label: string; mu: number[] }[] }> = {
  two: { label: "K=2 두 집단", K: 2, range: [-1.5, 6], inits: [{ label: "초기값 1", mu: [0, 1] }] },
  three: { label: "K=3 세 덩어리", K: 3, range: [-5.5, 3.5], inits: [{ label: "초기값 1", mu: [-2.72, -2.47, 0.18] }, { label: "초기값 2", mu: [-2.98, 0.54, 1.5] }] },
};
const initTheta = (mu: number[]): Theta => ({ pi: mu.map(() => 1 / mu.length), mu: [...mu], sd: mu.map(() => 1) });

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "GMM EM 단계", "stepper", narrow ? 0.95 : 0.66, "데이터 히스토그램과 성분 곡선, 책임도로 칠한 점, 단계별 로그우도와 ELBO");
  let dkey: DataKey = "two";
  let x = makeData(dkey);
  let th: Theta = initTheta(DATA.two.inits[0].mu);
  let q: number[][] = [];
  let hist: { ll: number; elbo: number; kind: "0" | "E" | "M" }[] = [];
  let iter = 0, seed = 0;
  const jit = (() => { const r = rng(7); return Array.from({ length: 600 }, () => r.u()); })();

  function reset(mu: number[]) {
    th = initTheta(mu);
    const K = mu.length;
    q = x.map(() => Array(K).fill(1 / K));
    iter = 0;
    hist = [{ ll: logLik(x, th), elbo: elbo(x, q, th), kind: "0" }];
  }
  const push = (kind: "E" | "M") => { hist.push({ ll: logLik(x, th), elbo: elbo(x, q, th), kind }); };
  const doE = () => { q = eStep(x, th); push("E"); };
  const doM = () => { th = mStep(x, q); iter++; push("M"); };

  // 성분 색: 파랑(--chart-3), 초록(--success), 글자색(--foreground). 점은 q(z_n)로 이 색들을 섞는다.
  const comps = () => [C.edge, C.good, C.ink];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const D = DATA[dkey], [X0, X1] = D.range, K = D.K;
    const pad = 16, top = 12;
    const split = Math.round(H * 0.6);
    const stripH = 26, base = split - stripH - 22;
    const X = (v: number) => pad + ((v - X0) / (X1 - X0)) * (W - 2 * pad);
    // 히스토그램(밀도)
    const bw = 0.2, nb = Math.ceil((X1 - X0) / bw), cnt = new Array(nb).fill(0);
    x.forEach((v) => { const b = Math.floor((v - X0) / bw); if (b >= 0 && b < nb) cnt[b]++; });
    const dens = cnt.map((c) => c / (x.length * bw));
    const mix = (v: number) => th.pi.reduce((s, p, k) => s + p * Math.exp(-0.5 * ((v - th.mu[k]) / th.sd[k]) ** 2) / (th.sd[k] * Math.sqrt(2 * Math.PI)), 0);
    const grid = Array.from({ length: 241 }, (_, i) => X0 + ((X1 - X0) * i) / 240);
    const ymax = Math.max(Math.max(...dens), ...grid.map((v) => Math.min(mix(v), 2.5))) * 1.1;
    const Y = (v: number) => base - (Math.min(v, ymax) / ymax) * (base - top);
    ctx.fillStyle = C.faint;
    dens.forEach((d, b) => { if (d > 0) ctx.fillRect(X(X0 + b * bw) + 0.5, Y(d), Math.max(1, X(X0 + (b + 1) * bw) - X(X0 + b * bw) - 1), base - Y(d)); });
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, base); ctx.lineTo(W - pad, base); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let k = Math.ceil(X0); k <= X1; k++) { const t = String(k); ctx.fillText(t, X(k) - ctx.measureText(t).width / 2, base + 13); }
    // 성분 곡선과 혼합 밀도
    const cols = comps();
    for (let k = 0; k < K; k++) {
      ctx.strokeStyle = cols[k]; ctx.lineWidth = 2;
      ctx.beginPath();
      grid.forEach((v, i) => { const y = Y(th.pi[k] * Math.exp(-0.5 * ((v - th.mu[k]) / th.sd[k]) ** 2) / (th.sd[k] * Math.sqrt(2 * Math.PI))); if (i) ctx.lineTo(X(v), y); else ctx.moveTo(X(v), y); });
      ctx.stroke();
      ctx.fillStyle = cols[k];
      ctx.beginPath(); ctx.moveTo(X(th.mu[k]), base - 1); ctx.lineTo(X(th.mu[k]) - 5, base + 7); ctx.lineTo(X(th.mu[k]) + 5, base + 7); ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
    ctx.beginPath(); grid.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(mix(v))) : ctx.moveTo(X(v), Y(mix(v))))); ctx.stroke();
    ctx.setLineDash([]);
    // 점: q(z_n)로 성분 색을 섞는다
    const rgbs = comps().map(rgbOf);
    const sy = base + 18;
    x.forEach((v, n) => {
      if (v < X0 || v > X1) return;
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < K; k++) { r += q[n][k] * rgbs[k][0]; g += q[n][k] * rgbs[k][1]; b += q[n][k] * rgbs[k][2]; }
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.beginPath(); ctx.arc(X(v), sy + jit[n % jit.length] * (stripH - 6) + 3, 2.2, 0, 7); ctx.fill();
    });
    // 아래: 단계별 log p(x)와 ELBO. 꼭대기 근처는 선형, 멀리 떨어진 값은 압축하는 단조 변환(asinh)으로 그린다.
    const t0 = split + 18, t1 = H - 22;
    const left = pad + (narrow ? 40 : 48), right = W - pad;
    const hiV = Math.max(...hist.map((h) => h.ll)), loV = Math.min(...hist.map((h) => h.elbo));
    const s = 3, span = Math.asinh((hiV - loV) / s) || 1;
    const TY = (v: number) => t0 + 6 + (Math.asinh((hiV - v) / s) / span) * (t1 - t0 - 6);
    const nShow = Math.max(12, hist.length);
    const TX = (i: number) => left + (i / (nShow - 1)) * (right - left - 6);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, t0); ctx.lineTo(left, t1); ctx.lineTo(right, t1); ctx.stroke();
    ctx.fillStyle = C.mid;
    [...new Set([hiV, loV].map((v) => Math.round(v)))].forEach((v) => { const t = String(v); ctx.fillText(t, left - 6 - ctx.measureText(t).width, Math.min(t1, Math.max(t0 + 8, TY(v) + 4))); });
    ctx.fillText("단계", right - ctx.measureText("단계").width, t1 + 14);
    // ELBO(속이 빈 점)와 log p(선)
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); hist.forEach((h, i) => (i ? ctx.lineTo(TX(i), TY(h.ll)) : ctx.moveTo(TX(i), TY(h.ll)))); ctx.stroke();
    hist.forEach((h, i) => {
      if (h.ll - h.elbo > 1e-6) { ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(TX(i), TY(h.ll)); ctx.lineTo(TX(i), TY(h.elbo)); ctx.stroke(); ctx.setLineDash([]); }
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(TX(i), TY(h.elbo), 4, 0, 7); ctx.stroke();
      ctx.fillStyle = C.ink;
      ctx.beginPath(); ctx.arc(TX(i), TY(h.ll), 2.6, 0, 7); ctx.fill();
    });
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const lw = ctx.measureText("log p(x)  ").width, ew = ctx.measureText("ELBO").width;
    ctx.fillStyle = C.ink; ctx.fillText("log p(x)", right - lw - ew, t1 - 10);
    ctx.fillStyle = C.edge; ctx.fillText("ELBO", right - ew, t1 - 10);
  }

  function update() {
    const f2 = (a: number[]) => `(${a.map((v) => v.toFixed(2)).join(", ")})`;
    const h = hist[hist.length - 1];
    const drops = hist.filter((g, i) => i > 0 && g.ll < hist[i - 1].ll - 1e-9).length;
    ro.set("ll", `반복 ${iter} · log p(x) ${h.ll.toFixed(2)}`, "hot");
    ro.set("elbo", `ELBO ${h.elbo.toFixed(2)}`);
    ro.set("kl", `KL 갭 ${Math.max(0, h.ll - h.elbo).toFixed(2)}`, h.ll - h.elbo < 1e-6 ? "ok" : "");
    ro.set("drop", `log p 감소 ${drops}번`, drops === 0 ? "ok" : "hot");
    ro.set("mu", `μ ${f2(th.mu)}`);
    draw();
  }

  const dataBtns = presets((Object.keys(DATA) as DataKey[]).map((k) => ({ key: k, label: DATA[k].label })), (k) => {
    dkey = k; x = makeData(k); buildInits(); reset(DATA[k].inits[0].mu); update();
  }, dkey);
  dataBtns.row.classList.add("segmented");
  const initWrap = el("div");
  let initBtns: ReturnType<typeof presets<string>>;
  function buildInits() {
    initBtns = presets(DATA[dkey].inits.map((it, i) => ({ key: String(i), label: it.label })), (k) => { reset(DATA[dkey].inits[Number(k)].mu); update(); }, "0");
    initWrap.replaceChildren(initBtns.row);
  }
  buildInits();
  const tools = el("div", "wframe-row");
  const btn = (label: string, fn: () => void) => { const b = el("button", "preset"); b.type = "button"; b.textContent = label; b.addEventListener("click", fn); tools.append(b); return b; };
  btn("E-step", () => { doE(); update(); });
  btn("M-step", () => { doM(); update(); });
  btn("EM ×10", () => { for (let i = 0; i < 10; i++) { doE(); doM(); } update(); });
  btn("↻ 무작위 초기값", () => {
    const r = rng(101 + seed++);
    const K = DATA[dkey].K, idx: number[] = [];
    while (idx.length < K) { const i = Math.floor(r.u() * x.length); if (!idx.includes(i)) idx.push(i); }
    initBtns.select(null as unknown as string);
    reset(idx.map((i) => Math.round(x[i] * 100) / 100)); update();
  });
  const ro = readouts();
  const top = el("div");
  top.style.cssText = "display:flex;flex-wrap:wrap;gap:8px";
  top.append(dataBtns.row, initWrap);
  f.panel.append(top, tools, ro.row);
  f.onResize(draw);
  reset(DATA.two.inits[0].mu);
  update();
}
