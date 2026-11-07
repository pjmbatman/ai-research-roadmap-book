// 밴딧 누적 후회 (data/widgets.toml: bandit-regret).
// 20.4장의 예: 성공 확률 0.30·0.45·0.50·0.55·0.60인 Bernoulli 팔 5개, T = 2000 step, 실행 200번의 평균 누적 (기대) 후회.
// 팔 간격 배율 s로 최선 팔(0.60)과의 격차를 s배 한다: p_a = 0.60 − s·(0.60 − p_a⁰). s = 1이 장의 예다.
// 방법은 장의 코드와 같다: 무작위, ε-greedy(ε = 0.1, 안 해 본 팔 먼저), UCB1(처음 K step은 팔마다 한 번, 그다음 ŝ/n + √(2 ln t / n)),
// Thompson(Beta(1 + 성공, 1 + 실패)에서 하나씩 뽑기). 실행 r의 난수는 고정 시드라 간격을 바꿔도 같은 난수열을 쓴다.
// 실행 200번은 여러 프레임에 나눠 계산하고, 도는 동안 지금까지의 평균을 그린다.
import { frame, param, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

export const BASE = [0.30, 0.45, 0.50, 0.55, 0.60], K = 5, T = 2000, RUNS = 200;
export type Algo = "random" | "eps" | "ucb" | "ts";
export const ALGOS: { key: Algo; name: string }[] = [
  { key: "random", name: "무작위" },
  { key: "eps", name: "ε-greedy(0.1)" },
  { key: "ucb", name: "UCB1" },
  { key: "ts", name: "Thompson" },
];
export const armsFor = (s: number) => BASE.map((p) => 0.6 - s * (0.6 - p));

type R = ReturnType<typeof rng>;
/** Gamma(shape ≥ 1, 1): Marsaglia–Tsang */
function gamma(r: R, a: number) {
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do { x = r.normal(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r.u();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(Math.max(u, 1e-300)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
const betaDraw = (r: R, a: number, b: number) => { const x = gamma(r, a); return x / (x + gamma(r, b)); };

/** 한 실행의 step별 기대 후회 μ* − μ_{a_t}를 inc에 더한다 */
export function runOnce(algo: Algo, p: number[], seed: number, inc: Float64Array) {
  const r = rng(seed);
  const n = new Float64Array(K), s = new Float64Array(K);
  const best = Math.max(...p);
  for (let t = 0; t < T; t++) {
    let a = 0;
    if (algo === "random") a = Math.floor(r.u() * K);
    else if (algo === "eps") {
      if (r.u() < 0.1) a = Math.floor(r.u() * K);
      else { let m = -1; for (let i = 0; i < K; i++) { const v = n[i] > 0 ? s[i] / n[i] : 1; if (v > m) { m = v; a = i; } } }
    } else if (algo === "ucb") {
      if (t < K) a = t;
      else { let m = -Infinity; for (let i = 0; i < K; i++) { const v = s[i] / n[i] + Math.sqrt((2 * Math.log(t)) / n[i]); if (v > m) { m = v; a = i; } } }
    } else {
      let m = -1; for (let i = 0; i < K; i++) { const v = betaDraw(r, 1 + s[i], 1 + n[i] - s[i]); if (v > m) { m = v; a = i; } }
    }
    const win = r.u() < p[a] ? 1 : 0;
    n[a] += 1; s[a] += win;
    inc[t] += best - p[a];
  }
}

/** RUNS번 평균한 누적 후회 (검산용 동기 계산) */
export function cumRegret(algo: Algo, p: number[], runs = RUNS) {
  const inc = new Float64Array(T);
  for (let k = 0; k < runs; k++) runOnce(algo, p, seedOf(algo, k), inc);
  const out = new Float64Array(T);
  let c = 0; for (let t = 0; t < T; t++) { c += inc[t] / runs; out[t] = c; }
  return out;
}
const seedOf = (algo: Algo, k: number) => 1000 * (ALGOS.findIndex((x) => x.key === algo) + 1) + k;

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "밴딧 누적 후회", "simulate", narrow ? 0.85 : 0.55, "step t에 따른 네 방법의 평균 누적 후회 곡선");
  let gap = 1;
  const on: Record<Algo, boolean> = { random: true, eps: true, ucb: true, ts: true };
  const inc: Record<Algo, Float64Array> = { random: new Float64Array(T), eps: new Float64Array(T), ucb: new Float64Array(T), ts: new Float64Array(T) };
  const cum: Record<Algo, Float64Array> = { random: new Float64Array(T), eps: new Float64Array(T), ucb: new Float64Array(T), ts: new Float64Array(T) };
  let done = 0, job = 0;
  const color = (a: Algo) => (a === "ts" ? C.edge : a === "ucb" ? C.ink2 : a === "eps" ? C.good : C.mid);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 40, PR = narrow ? 78 : 104, top = 30, bottom = H - 26;
    const shown = ALGOS.filter((a) => on[a.key] && a.key !== "random");
    let ymax = 10;
    for (const a of shown) ymax = Math.max(ymax, cum[a.key][T - 1]);
    if (!shown.length) ymax = Math.max(ymax, cum.random[T - 1]);
    ymax *= 1.12;
    const tick = [5, 10, 20, 25, 50, 100, 200, 250, 500].find((s) => ymax / s <= 5) ?? 1000;
    const X = (t: number) => PL + (t / T) * (W - PL - PR);
    const Y = (v: number) => bottom - (v / ymax) * (bottom - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (let v = 0; v <= ymax; v += tick) {
      ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke();
      ctx.fillText(String(v), PL - 6 - ctx.measureText(String(v)).width, Y(v) + 4);
    }
    for (const t of [0, 500, 1000, 1500, 2000]) { const s = String(t); ctx.fillText(s, X(t) - ctx.measureText(s).width / 2, bottom + 15); }
    ctx.fillText("누적 후회", PL, top - 12);
    const lab = `실행 ${done}/${RUNS}번 평균`;
    ctx.fillText(lab, W - PR - ctx.measureText(lab).width, top - 12);
    ctx.save();
    ctx.beginPath(); ctx.rect(PL, top - 2, W - PL - PR + 2, bottom - top + 4); ctx.clip();
    const ends: { y: number; a: Algo }[] = [];
    for (const { key } of ALGOS) {
      if (!on[key] || !done) continue;
      const c = cum[key];
      ctx.strokeStyle = color(key); ctx.lineWidth = key === "random" ? 1.4 : 2.2;
      if (key === "random") ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let t = 0; t < T; t += 4) { if (t) ctx.lineTo(X(t + 1), Y(c[t])); else ctx.moveTo(X(t + 1), Y(c[t])); }
      ctx.lineTo(X(T), Y(c[T - 1]));
      ctx.stroke(); ctx.setLineDash([]);
      ends.push({ y: Math.max(top + 4, Math.min(bottom, Y(c[T - 1]))), a: key });
    }
    ctx.restore();
    // 곡선 끝에 이름표(겹치지 않게 아래로 민다)
    ends.sort((p, q) => p.y - q.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    for (const e of ends) {
      const name = ALGOS.find((x) => x.key === e.a)!.name;
      ctx.fillStyle = color(e.a);
      ctx.fillText(narrow && e.a === "eps" ? "ε-greedy" : name, W - PR + 6, Math.min(bottom, e.y) + 4);
    }
  }

  function readout() {
    const p = armsFor(gap);
    ro.set("arms", `팔 μ ${p.map((v) => v.toFixed(2)).join(" · ")}`);
    for (const { key, name } of ALGOS) {
      if (key === "random") continue;
      const c = cum[key];
      const short = key === "eps" ? "ε-greedy" : name;
      ro.set(key, on[key] && done ? `${short} ${c[T - 1].toFixed(1)} (뒤 1000 +${(c[T - 1] - c[999]).toFixed(1)})` : `${short} 꺼짐`, key === "ts" && on[key] ? "hot" : "");
    }
    pageState.set("bandit", { gap, regret: Object.fromEntries(ALGOS.map(({ key }) => [key, cum[key][T - 1]])) }, root);
  }

  function accumulate() {
    for (const { key } of ALGOS) {
      let c = 0;
      const a = inc[key], o = cum[key];
      for (let t = 0; t < T; t++) { c += a[t] / done; o[t] = c; }
    }
  }

  // 간격이 바뀌면 처음부터 다시: 프레임마다 실행 몇 번씩
  function restart() {
    const my = ++job;
    for (const { key } of ALGOS) inc[key].fill(0);
    done = 0;
    const p = armsFor(gap);
    const step = () => {
      if (my !== job) return;
      const t0 = performance.now();
      while (done < RUNS && performance.now() - t0 < 14) {
        for (const { key } of ALGOS) runOnce(key, p, seedOf(key, done), inc[key]);
        done++;
      }
      accumulate();
      readout();
      draw();
      if (done < RUNS) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  const pg = param({ label: "간격", min: 0.25, max: 2, step: 0.05, value: gap, fmt: (v) => v.toFixed(2), aria: "최선 팔과의 격차 배율", onInput: (v) => { gap = v; restart(); } });
  const grid = el("div", "param-grid");
  grid.append(pg.row);
  const tools = el("div", "wframe-row");
  for (const { key, name } of ALGOS) {
    if (key === "random") continue;
    const lb = el("label", "toggle");
    const inp = el("input");
    inp.type = "checkbox"; inp.checked = true;
    inp.addEventListener("change", () => { on[key] = inp.checked; readout(); draw(); });
    lb.append(inp, document.createTextNode(` ${name}`));
    tools.append(lb);
  }
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);
  f.onResize(draw);
  restart();
}
