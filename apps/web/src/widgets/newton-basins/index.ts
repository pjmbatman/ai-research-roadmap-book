// Newton 끌개 영역 (data/widgets.toml: newton-basins).
// 복소 다항식 p(z) = 0을 Newton 방법 z ← z − t·p(z)/p′(z)로 푼다. 실수 두 변수로 쓰면 9.4장 실습의 2×2 Newton과 같다
// (Jacobian의 행렬식이 |p′(z)|²). [−2, 2]² 격자(linspace, 끝점 포함)의 각 점에서 출발해 어느 근에 도착했는지로 칠하고,
// 반복 수가 많을수록 흐리게 칠한다. 수렴하지 못한 점은 진한 색이다.
// 판정은 labs/numerical-methods/nonlinear-equations/newton_basins.py와 같다: 잔차 |p| < 1e-10이면 멈추고, 끝난 뒤 근과의 거리 < 1e-6이면 그 근.
// damping은 |p|가 줄어들 때까지 t를 1, 1/2, 1/4, …로 최대 30번 줄인다. 계산은 몇 행씩 나눠 프레임마다 이어 한다.
import { frame, param, presets, readouts, el, C, rgbOf } from "../frame";

type Cx = [number, number];
interface Poly { label: string; coef: number[]; guess: Cx[] } // coef: 높은 차수부터
export const POLYS: Record<string, Poly> = {
  cube: { label: "z³ − 1", coef: [1, 0, 0, -1], guess: [[1, 0], [-0.5, 0.866], [-0.5, -0.866]] },
  quart: { label: "z⁴ − 1", coef: [1, 0, 0, 0, -1], guess: [[1, 0], [0, 1], [-1, 0], [0, -1]] },
  cycle: { label: "z³ − 2z + 2", coef: [1, 0, -2, 2], guess: [[-1.77, 0], [0.885, 0.59], [0.885, -0.59]] },
};

/** p(z)와 p′(z)를 Horner로 함께 계산한다. */
function horner(c: number[], x: number, y: number) {
  let pr = c[0], pi = 0, dr = 0, di = 0;
  for (let k = 1; k < c.length; k++) {
    const ndr = dr * x - di * y + pr, ndi = dr * y + di * x + pi;
    dr = ndr; di = ndi;
    const npr = pr * x - pi * y + c[k], npi = pr * y + pi * x;
    pr = npr; pi = npi;
  }
  return [pr, pi, dr, di];
}
/** 대략의 근을 Newton으로 다듬는다. */
export function roots(P: Poly): Cx[] {
  return P.guess.map(([x, y]) => {
    for (let i = 0; i < 60; i++) {
      const [pr, pi, dr, di] = horner(P.coef, x, y), d = dr * dr + di * di;
      x -= (pr * dr + pi * di) / d; y -= (pi * dr - pr * di) / d;
    }
    return [x, y] as Cx;
  });
}

/** 한 초기값에서 Newton을 돌린다. which = 근 번호(실패면 −1), iters = 멈춘 반복(못 멈추면 maxIter) */
export function newton(P: Poly, R: Cx[], x0: number, y0: number, maxIter: number, damped: boolean, path?: Cx[]) {
  let x = x0, y = y0, iters = maxIter;
  path?.push([x, y]);
  for (let k = 0; k < maxIter; k++) {
    const [pr, pi, dr, di] = horner(P.coef, x, y);
    const res = Math.hypot(pr, pi);
    if (res < 1e-10) { iters = k; break; }
    let det = dr * dr + di * di;
    if (Math.abs(det) < 1e-300) det = 1e-300;                     // 특이 Jacobian 보호(실습 코드와 같다)
    const sx = (pr * dr + pi * di) / det, sy = (pi * dr - pr * di) / det;   // p/p′
    let t = 1;
    if (damped) {
      for (let q = 0; q < 30; q++) {
        const [gr, gi] = horner(P.coef, x - t * sx, y - t * sy);
        if (Math.hypot(gr, gi) >= res) t /= 2; else break;
      }
    }
    x -= t * sx; y -= t * sy;
    path?.push([x, y]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) break;
  }
  let which = -1, best = 1e-6;
  R.forEach(([rx, ry], i) => { const d = Math.hypot(x - rx, y - ry); if (d < best) { best = d; which = i; } });
  return { which, iters };
}

/** 격자 전체의 통계(검산용) */
export function gridStats(key: string, n: number, maxIter: number, damped: boolean) {
  const P = POLYS[key], R = roots(P);
  const cnt = new Array(R.length + 1).fill(0);
  let sum = 0, mx = 0, ok = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const { which, iters } = newton(P, R, -2 + (4 * i) / (n - 1), -2 + (4 * j) / (n - 1), maxIter, damped);
    cnt[which + 1]++;
    if (which >= 0) { ok++; sum += iters; mx = Math.max(mx, iters); }
  }
  return { share: cnt.slice(1).map((c) => c / (n * n)), fail: cnt[0] / (n * n), mean: sum / ok, max: mx };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 480;
  const f = frame(root, "Newton 끌개 영역", "simulate", narrow ? 1.0 : 0.62, "복소평면 [−2, 2]²의 각 초기값이 Newton 반복으로 도착하는 근을 색으로 칠한 지도와 고른 초기값의 반복 경로");
  const RES = narrow ? 160 : 220;
  let key = attrs.preset in POLYS ? attrs.preset : "cube";
  let P = POLYS[key], R = roots(P);
  let damped = false, maxIter = 50;
  let start: Cx = [-1.14, 0.2];
  let which = new Int8Array(RES * RES), iters = new Uint8Array(RES * RES);
  let row = 0, job = 0, dragging = false;
  const off = document.createElement("canvas");
  off.width = off.height = RES;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(RES, RES);

  const palette = () => [C.edge, C.ink2, C.good, C.mid].map(rgbOf);
  function paintRows(j0: number, j1: number) {
    const pal = palette(), bg = rgbOf(C.bg), ink = rgbOf(C.ink);
    for (let j = j0; j < j1; j++) for (let i = 0; i < RES; i++) {
      const k = j * RES + i, w = which[k];
      // 그림의 위쪽이 y = +2: 격자 행 j(y 오름차순)를 화면 행 RES − 1 − j에 둔다.
      const o = 4 * ((RES - 1 - j) * RES + i);
      let c: number[];
      if (w < 0) c = ink;
      else { const s = 0.65 * Math.min(1, iters[k] / 20), base = pal[w % pal.length]; c = base.map((v, q) => v + (bg[q] - v) * s); }
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  function compute() {
    const my = ++job;
    row = 0;
    const step = () => {
      if (my !== job) return;
      const t0 = performance.now();
      const r0 = row;
      while (row < RES && performance.now() - t0 < 10) {
        for (let i = 0; i < RES; i++) {
          const res = newton(P, R, -2 + (4 * i) / (RES - 1), -2 + (4 * row) / (RES - 1), maxIter, damped);
          which[row * RES + i] = res.which; iters[row * RES + i] = res.iters;
        }
        row++;
      }
      paintRows(r0, row);
      octx.putImageData(img, 0, 0);
      draw();
      if (row < RES) requestAnimationFrame(step); else stats();
    };
    requestAnimationFrame(step);
  }

  const geom = () => { const side = Math.min(f.width - 16, f.height - 16); return { side, ox: (f.width - side) / 2, oy: (f.height - side) / 2 }; };
  const toPx = ([x, y]: Cx): Cx => { const g = geom(); return [g.ox + ((x + 2) / 4) * g.side, g.oy + ((2 - y) / 4) * g.side]; };
  const toW = (px: number, py: number): Cx => { const g = geom(); return [((px - g.ox) / g.side) * 4 - 2, 2 - ((py - g.oy) / g.side) * 4]; };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const g = geom();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, g.ox, g.oy, g.side, g.side);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(g.ox, g.oy, g.side, g.side);
    // 근
    for (const r of R) {
      const q = toPx(r);
      ctx.beginPath(); ctx.moveTo(q[0], q[1] - 7); ctx.lineTo(q[0] + 7, q[1]); ctx.lineTo(q[0], q[1] + 7); ctx.lineTo(q[0] - 7, q[1]); ctx.closePath();
      ctx.fillStyle = C.ink; ctx.fill();
      ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5; ctx.stroke();
    }
    // 고른 초기값의 경로
    const path: Cx[] = [];
    newton(P, R, start[0], start[1], maxIter, damped, path);
    ctx.save();
    ctx.beginPath(); ctx.rect(g.ox, g.oy, g.side, g.side); ctx.clip();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.lineJoin = "round";
    ctx.beginPath();
    path.forEach((p, i) => { const q = toPx([Math.max(-50, Math.min(50, p[0])), Math.max(-50, Math.min(50, p[1]))]); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
    ctx.stroke();
    ctx.fillStyle = C.ink;
    for (const p of path.slice(1)) { const q = toPx(p); ctx.beginPath(); ctx.arc(q[0], q[1], 2.4, 0, 7); ctx.fill(); }
    ctx.restore();
    const s = toPx(start);
    ctx.fillStyle = C.bg; ctx.beginPath(); ctx.arc(s[0], s[1], dragging ? 8 : 7, 0, 7); ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.4; ctx.stroke();
    if (row < RES) {
      ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard, sans-serif";
      ctx.fillText(`계산 중 ${Math.round((100 * row) / RES)}%`, g.ox + 8, g.oy + 18);
    }
  }

  function startReadout() {
    const r = newton(P, R, start[0], start[1], maxIter, damped);
    ro.set("start", `z₀ = (${start[0].toFixed(2)}, ${start[1].toFixed(2)}) → ${r.which >= 0 ? `근${r.which + 1} (${r.iters}번)` : "수렴 실패"}`, "hot");
  }
  function stats() {
    const n = RES * RES, cnt = new Array(R.length + 1).fill(0);
    let sum = 0, mx = 0, ok = 0;
    for (let k = 0; k < n; k++) { cnt[which[k] + 1]++; if (which[k] >= 0) { ok++; sum += iters[k]; mx = Math.max(mx, iters[k]); } }
    ro.set("share", cnt.slice(1).map((c, i) => `근${i + 1} ${((100 * c) / n).toFixed(1)}%`).join(" · "));
    ro.set("fail", `실패 ${((100 * cnt[0]) / n).toFixed(2)}%`, cnt[0] > 0 ? "hot" : "ok");
    ro.set("iter", ok ? `평균 반복 ${(sum / ok).toFixed(2)} · 최대 ${mx}` : "수렴한 점 없음");
    ro.set("grid", `격자 ${RES}²`);
  }
  function recompute() { startReadout(); compute(); }

  const pre = presets(Object.entries(POLYS).map(([k, v]) => ({ key: k, label: v.label })), (k) => {
    key = k; P = POLYS[k]; R = roots(P); recompute();
  }, key);
  const tools = el("div", "wframe-row");
  const damp = el("label", "toggle");
  damp.innerHTML = `<input type="checkbox" /> damping`;
  damp.querySelector("input")!.addEventListener("change", (e) => { damped = (e.target as HTMLInputElement).checked; recompute(); });
  tools.append(damp);
  const pi = param({ label: "반복", min: 5, max: 50, step: 1, value: maxIter, fmt: (v) => `${v}`, aria: "최대 반복 수", onInput: (v) => { maxIter = v; recompute(); } });
  pi.row.style.gridTemplateColumns = "2.6em minmax(0, 1fr) 2.4em";
  const grid = el("div", "param-grid");
  grid.append(pi.row);
  const ro = readouts();
  f.panel.append(pre.row, tools, grid, ro.row);

  // 초기값 끌기
  const pos = (e: PointerEvent): Cx => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const setStart = (q: Cx) => {
    const w = toW(...q), cl = (v: number) => Math.max(-2, Math.min(2, Math.round(v * 100) / 100));
    start = [cl(w[0]), cl(w[1])];
    startReadout(); draw();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); setStart(pos(e)); });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) setStart(pos(e)); });
  const end = () => { dragging = false; draw(); };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);
  f.canvas.tabIndex = 0;
  f.canvas.addEventListener("keydown", (e) => {
    const d: Record<string, Cx> = { ArrowLeft: [-0.01, 0], ArrowRight: [0.01, 0], ArrowUp: [0, 0.01], ArrowDown: [0, -0.01] };
    if (!(e.key in d)) return;
    e.preventDefault();
    const cl = (v: number) => Math.max(-2, Math.min(2, Math.round(v * 100) / 100));
    start = [cl(start[0] + d[e.key][0]), cl(start[1] + d[e.key][1])];
    startReadout(); draw();
  });
  window.addEventListener("widget-recolor", () => { paintRows(0, row); octx.putImageData(img, 0, 0); draw(); });
  f.onResize(draw);
  recompute();
}
