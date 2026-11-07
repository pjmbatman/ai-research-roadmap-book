// 스펙트럼 합성곱과 해상도 (data/widgets.toml: spectral-conv).
// 24.4장 셀 3과 같은 비교: 입력 함수(파수 12까지, 계수 ~ (N(0,1) + i N(0,1)) / k^1.5)를 격자 N점에서 샘플링하고,
// 스펙트럼 합성곱(û = DFT(u)/N, 낮은 K개 모드에 복소 가중치 w_k를 곱하고 나머지는 버린 뒤 역변환)과
// 격자 칸 단위 3점 합성곱(0.25, 0.5, 0.25)을 적용한다. 32점 결과와 겹치는 점(N/32칸마다)의 최대 차이를 잰다.
// 연속 곡선은 해석식이다: 스펙트럼 합성곱 출력은 2 Re Σ_{k≤K} w_k a_k e^{2πikx}, 격자 N의 3점 합성곱은 모드마다 0.5 + 0.5 cos(2πk/N)을 곱한다.
import { frame, param, readouts, el, C, rng } from "../frame";

const K_IN = 12, K_MAX = 16;
type Cx = [number, number];

export function sample(seed: number) {
  const r = rng(seed);
  const a: Cx[] = Array.from({ length: K_IN }, (_, i) => { const k = (i + 1) ** 1.5; return [r.normal() / k, r.normal() / k]; });
  const w: Cx[] = Array.from({ length: K_MAX }, () => [r.normal(), r.normal()]);
  return { a, w };
}
/** 계수 → 격자 [0, 1)의 N점 값: 2 Re Σ a_k e^{2πikx} (본문 on_grid) */
export function onGrid(a: Cx[], N: number) {
  return Array.from({ length: N }, (_, n) => {
    let s = 0;
    a.forEach(([re, im], i) => { const th = (2 * Math.PI * (i + 1) * n) / N; s += 2 * (re * Math.cos(th) - im * Math.sin(th)); });
    return s;
  });
}
/** 본문 spectral_conv: rfft(u)/N의 1..K 모드에 w를 곱하고 irfft(·N). DFT를 직접 계산한다. */
export function spectralConv(u: number[], w: Cx[], K: number) {
  const N = u.length, kmax = Math.min(K, Math.floor(N / 2));
  const out = new Array<number>(N).fill(0);
  for (let k = 1; k <= kmax; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) { const th = (2 * Math.PI * k * n) / N; re += u[n] * Math.cos(th); im -= u[n] * Math.sin(th); }
    re /= N; im /= N;
    const [wr, wi] = w[k - 1];
    const pr = wr * re - wi * im;
    let pi = wr * im + wi * re;
    const mult = 2 * k === N ? 1 : 2; // 나이퀴스트 모드는 한 번만, 허수부는 버린다(irfft와 같다)
    if (2 * k === N) pi = 0;
    for (let n = 0; n < N; n++) { const th = (2 * Math.PI * k * n) / N; out[n] += mult * (pr * Math.cos(th) - pi * Math.sin(th)); }
  }
  return out;
}
export const stencilConv = (u: number[]) => u.map((v, n) => 0.25 * u[(n - 1 + u.length) % u.length] + 0.5 * v + 0.25 * u[(n + 1) % u.length]);
/** N점 결과와 32점 결과의 겹치는 점에서 최대 차이 */
export function compare(a: Cx[], w: Cx[], K: number, N: number) {
  const u32 = onGrid(a, 32), uN = onGrid(a, N), step = N / 32;
  const s32 = spectralConv(u32, w, K), sN = spectralConv(uN, w, K);
  const c32 = stencilConv(u32), cN = stencilConv(uN);
  let ds = 0, dc = 0;
  for (let i = 0; i < 32; i++) { ds = Math.max(ds, Math.abs(sN[i * step] - s32[i])); dc = Math.max(dc, Math.abs(cN[i * step] - c32[i])); }
  return { sN, cN, s32, c32, ds, dc };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "스펙트럼 합성곱과 해상도", "manipulate", narrow ? 1.15 : 0.66, "입력 함수, 스펙트럼 합성곱 출력, 3점 합성곱 출력을 격자 N과 32점 기준으로 비교한 그림");
  let K = Number(attrs.modes) || 12, N = 128, seed = 3;
  let { a, w } = sample(seed);
  let cmp = compare(a, w, K, N);
  const FINE = 400;
  const xs = Array.from({ length: FINE + 1 }, (_, i) => i / FINE);
  const series = (mult: (k: number) => Cx) => xs.map((x) => {
    let s = 0;
    a.forEach(([re, im], i) => { const k = i + 1, [mr, mi] = mult(k), pr = mr * re - mi * im, pi = mr * im + mi * re, th = 2 * Math.PI * k * x; s += 2 * (pr * Math.cos(th) - pi * Math.sin(th)); });
    return s;
  });

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const axL = 10, R = 10;
    const X = (x: number) => axL + x * (W - axL - R);
    const gap = 8, head = 20;
    const ph = (H - 3 * head - 2 * gap - 6) / 3;
    const panels = [0, 1, 2].map((i) => ({ t: i * (ph + head + gap) + head, b: i * (ph + head + gap) + head + ph }));
    const input = series(() => [1, 0]);
    const uN = onGrid(a, N);
    // 격자 값 v(길이 L)를 x = j/L에 놓고 주기 경계로 x = 1까지 잇는다
    const grid = (v: number[]) => [...v, v[0]].map((y, j) => [j / v.length, y] as [number, number]);
    const drawPanel = (i: number, title: string, line: [number, number][], lineCol: string, dots: number[] | null, rings: number[] | null) => {
      const { t, b } = panels[i];
      const all = [...line.map((p) => p[1]), ...(rings ?? [])];
      const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 + 1e-9;
      const Y = (v: number) => b - ((v - lo + pad) / (hi - lo + 2 * pad)) * (b - t);
      ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
      ctx.fillText(title, 6, t - 6);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(axL, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
      ctx.strokeStyle = lineCol; ctx.lineWidth = 1.8;
      ctx.beginPath(); line.forEach(([x, y], j) => (j ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke();
      if (dots) {
        const r = N > 128 ? 1.3 : N > 64 ? 1.8 : 2.4;
        ctx.fillStyle = C.ink2;
        dots.forEach((y, j) => { ctx.beginPath(); ctx.arc(X(j / dots.length), Y(y), r, 0, 7); ctx.fill(); });
      }
      if (rings) {
        // 32점 결과(○)와 같은 위치의 N점 결과를 세로 막대로 잇는다: 길이가 곧 해상도에 따른 차이다
        const step = (line.length - 1) / rings.length;
        ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
        rings.forEach((y, j) => { const x = X(j / rings.length); ctx.beginPath(); ctx.moveTo(x, Y(y)); ctx.lineTo(x, Y(line[j * step][1])); ctx.stroke(); });
        ctx.lineWidth = 1.4;
        rings.forEach((y, j) => { ctx.beginPath(); ctx.arc(X(j / rings.length), Y(y), narrow ? 2.6 : 3.2, 0, 7); ctx.stroke(); });
      }
    };
    drawPanel(0, `입력 a(x) · 격자 ${N}점`, xs.map((x, j) => [x, input[j]]), C.mid, uN, null);
    drawPanel(1, `스펙트럼 합성곱 K = ${K} · 선 ${N}점 · ○ 32점`, grid(cmp.sN), C.ink2, null, cmp.s32);
    drawPanel(2, `3점 합성곱 · 선 ${N}점 · ○ 32점`, grid(cmp.cN), C.ink2, null, cmp.c32);
  }

  function update() {
    cmp = compare(a, w, K, N);
    ro.set("s", `스펙트럼 K = ${K}: 32점과 차이 ${cmp.ds < 1e-12 ? "< 1e-12" : cmp.ds.toExponential(1)}`, cmp.ds < 1e-9 ? "ok" : "hot");
    ro.set("c", `3점: 32점과 차이 ${cmp.dc.toFixed(2)}`, N === 32 ? "ok" : "hot");
    draw();
  }

  const pk = param({ label: "K", min: 1, max: K_MAX, step: 1, value: K, fmt: (v) => `${v}`, aria: "남기는 Fourier 모드 수", onInput: (v) => { K = v; update(); } });
  const pn = param({ label: "N", min: 32, max: 256, step: 32, value: N, fmt: (v) => `${v}점`, aria: "격자 점 수", onInput: (v) => { N = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pk.row, pn.row);
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 새 함수·가중치";
  reseed.addEventListener("click", () => { seed++; ({ a, w } = sample(seed)); update(); });
  tools.append(reseed);
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);
  f.onResize(draw);
  update();
}
