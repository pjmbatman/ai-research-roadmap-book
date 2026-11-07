// 사영과 최소제곱 (data/widgets.toml: projection-lstsq, 7.4장 "사영"·"최소제곱").
// b를 A의 column space(직선 또는 평면)에 사영해 p = A x̂, 잔차 r = b − p를 그리고, Aᵀr = 0(잔차가 모든 열과 직교)을 숫자로 보인다.
// x̂는 정규방정식 AᵀA x̂ = Aᵀb의 해다. '후보 x' 슬라이더로 x를 x̂에서 옮기면 ‖b − Ax‖가 커지고 Aᵀ(b − Ax) ≠ 0이 된다.
// 조작: 직선(ℝ²)·평면(ℝ³, 본문 예 A = [[1,0],[1,1],[1,2]], b = (6,0,0)) 프리셋, b 성분 슬라이더, ℝ²에서는 b 끌기, ℝ³에서는 그림을 끌어 시점 회전.
import { frame, param, presets, readouts, el, C } from "../frame";

type Vec = number[];
const dotv = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);
const sub = (a: Vec, b: Vec) => a.map((v, i) => v - b[i]);
const add = (a: Vec, b: Vec) => a.map((v, i) => v + b[i]);
const scl = (a: Vec, s: number) => a.map((v) => v * s);
const norm = (a: Vec) => Math.sqrt(dotv(a, a));

/** 열 cols(각 길이 m)로 이루어진 A에 대해 최소제곱: 정규방정식(n = 1, 2)으로 x̂, p, r, Aᵀr */
export function lstsq(cols: Vec[], b: Vec) {
  const G = cols.map((ci) => cols.map((cj) => dotv(ci, cj))); // AᵀA
  const h = cols.map((c) => dotv(c, b));                        // Aᵀb
  let x: Vec;
  if (cols.length === 1) x = [h[0] / G[0][0]];
  else {
    const det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
    x = [(G[1][1] * h[0] - G[0][1] * h[1]) / det, (G[0][0] * h[1] - G[1][0] * h[0]) / det];
  }
  const p = Ax(cols, x);
  const r = sub(b, p);
  return { x, p, r, Atr: cols.map((c) => dotv(c, r)) };
}
export const Ax = (cols: Vec[], x: Vec) => cols.reduce((s, c, j) => add(s, scl(c, x[j])), cols[0].map(() => 0));

const MODES = {
  line: { label: "직선 위 (ℝ²)", cols: [[2, 1]], b: [3, 4] },
  plane: { label: "평면 위 (ℝ³, 본문 예)", cols: [[1, 1, 1], [0, 1, 2]], b: [6, 0, 0] },
} as const;
type Mode = keyof typeof MODES;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "사영과 최소제곱", "manipulate", root.clientWidth < 560 ? 0.95 : 0.6,
    "b와 그 사영 p, 잔차 r, A의 column space(직선 또는 평면)를 그린 그림");
  let mode: Mode = attrs.preset === "plane" ? "plane" : "line";
  let b: Vec = [...MODES[mode].b];
  let delta = 0;
  let az = -1.0, el3 = 0.55; // ℝ³ 시점(방위각, 고도각)
  let drag: null | "b" | "view" = null;
  let last: [number, number] = [0, 0];

  const cols = () => MODES[mode].cols.map((c) => [...c]) as Vec[];
  const dim = () => (mode === "line" ? 2 : 3);
  /** 후보 x = x̂ + δ·e_last (ℝ²: x 자체, ℝ³: 기울기 성분 x₂) */
  const candidate = (xh: Vec) => xh.map((v, i) => (i === xh.length - 1 ? v + delta : v));

  // ── 화면 좌표
  function proj3(v: Vec): [number, number, number] {
    const [x, y, z] = v;
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el3), se = Math.sin(el3);
    const X = ca * x - sa * y;
    const Y0 = sa * x + ca * y;
    return [X, ce * z - se * Y0, se * z + ce * Y0];
  }
  // 축척은 b에 따라 바꾸지 않는다(끌거나 슬라이더를 움직일 때 그림이 흔들리지 않게)
  const scaleOf = () => (Math.min(f.width, f.height) / 2 - 14) / (dim() === 2 ? 5.5 : 7.2);
  function toPx(v: Vec, s: number): [number, number] {
    if (v.length === 2) return [f.width / 2 + v[0] * s, f.height / 2 - v[1] * s];
    const [X, Y] = proj3(v);
    return [f.width / 2 + X * s, f.height / 2 - Y * s];
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const A = cols();
    const { x, p, r } = lstsq(A, b);
    const xc = candidate(x);
    const q = Ax(A, xc);
    const s = scaleOf();
    const P = (v: Vec) => toPx(v, s);
    ctx.lineWidth = 1;
    if (dim() === 2) {
      // 격자와 축
      ctx.strokeStyle = C.faint;
      const R = Math.ceil(Math.max(W, H) / s / 2) + 1;
      for (let k = -R; k <= R; k++) { line(P([k, -R]), P([k, R])); line(P([-R, k]), P([R, k])); }
      ctx.strokeStyle = C.mid;
      line(P([-R, 0]), P([R, 0])); line(P([0, -R]), P([0, R]));
      // column space = a 방향 직선
      const a = A[0], u = scl(a, 1 / norm(a));
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 3;
      line(P(scl(u, -2 * R)), P(scl(u, 2 * R)));
      ctx.lineWidth = 1;
      arrow(P([0, 0]), P(a), C.ink2, "a");
    } else {
      // 축
      const L = 3.2;
      ctx.strokeStyle = C.grid2;
      ([[L, 0, 0, "x₁"], [0, L, 0, "x₂"], [0, 0, L, "x₃"]] as [number, number, number, string][]).forEach(([X, Y, Z, n]) => {
        line(P([0, 0, 0]), P([X, Y, Z]));
        ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
        const t = P([X * 1.08, Y * 1.08, Z * 1.08]); ctx.fillText(n, t[0] - 6, t[1] + 4);
      });
      // 평면 조각: 정규직교 basis e1, e2 (Gram–Schmidt)
      const e1 = scl(A[0], 1 / norm(A[0]));
      const t2 = sub(A[1], scl(e1, dotv(e1, A[1])));
      const e2 = scl(t2, 1 / norm(t2));
      const S = Math.max(2.5, norm(p), norm(q));
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, j]) => P(add(scl(e1, i * S), scl(e2, j * S))));
      ctx.beginPath();
      corners.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
      ctx.closePath();
      ctx.fillStyle = C.fill; ctx.fill();
      ctx.strokeStyle = C.ink2; ctx.stroke();
      // 평면 위 격자선 몇 개
      ctx.strokeStyle = C.edgeSoft;
      for (let k = -2; k <= 2; k++) {
        const c = (k / 2) * S;
        line(P(add(scl(e1, c), scl(e2, -S))), P(add(scl(e1, c), scl(e2, S))));
        line(P(add(scl(e1, -S), scl(e2, c))), P(add(scl(e1, S), scl(e2, c))));
      }
      arrow(P([0, 0, 0]), P(A[0]), C.ink2, "a₁");
      arrow(P([0, 0, 0]), P(A[1]), C.ink2, "a₂");
    }
    const O = P(b.map(() => 0));
    // 후보 점 Ax (δ ≠ 0)
    if (Math.abs(delta) > 1e-9) {
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5;
      line(P(b), P(q));
      ctx.setLineDash([]);
      ctx.fillStyle = C.mid;
      dot(P(q), 5);
      ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText("Ax", P(q)[0] + 8, P(q)[1] + 14);
    }
    // b, p, r
    ctx.setLineDash([5, 4]); ctx.strokeStyle = C.mid; line(O, P(b)); ctx.setLineDash([]);
    // 직각 표시: p에서 r 방향과 column space 안의 방향
    if (norm(r) > 1e-6) {
      const rh = scl(r, 1 / norm(r));
      const inPlane = norm(p) > 1e-6 ? scl(p, -1 / norm(p)) : scl(A[0], 1 / norm(A[0]));
      const m = 0.32 * Math.max(3, norm(b)) / 4;
      const c1 = add(p, scl(rh, m)), c3 = add(p, scl(inPlane, m)), c2 = add(c1, scl(inPlane, m));
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(...P(c1)); ctx.lineTo(...P(c2)); ctx.lineTo(...P(c3)); ctx.stroke();
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6;
    line(P(p), P(b));
    ctx.lineWidth = 1;
    const mid = P(add(p, scl(r, 0.5)));
    ctx.fillStyle = C.edge; ctx.font = "700 13px JetBrains Mono, monospace";
    ctx.fillText("r", mid[0] + 8, mid[1] - 4);
    ctx.fillStyle = C.ink;
    dot(P(p), 5);
    ctx.fillText("p = Ax̂", P(p)[0] + 9, P(p)[1] + 16);
    ctx.fillStyle = C.edge;
    dot(P(b), drag === "b" ? 9 : 7);
    ctx.fillStyle = C.ink;
    ctx.fillText("b", P(b)[0] + 10, P(b)[1] - 8);
  }
  function line(a: [number, number], c: [number, number]) { f.ctx.beginPath(); f.ctx.moveTo(...a); f.ctx.lineTo(...c); f.ctx.stroke(); }
  function dot(p: [number, number], r: number) { f.ctx.beginPath(); f.ctx.arc(p[0], p[1], r, 0, 7); f.ctx.fill(); }
  function arrow(a: [number, number], c: [number, number], color: string, label: string) {
    const { ctx } = f;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
    line(a, c);
    const ang = Math.atan2(c[1] - a[1], c[0] - a[0]);
    ctx.beginPath();
    ctx.moveTo(c[0], c[1]);
    ctx.lineTo(c[0] - 10 * Math.cos(ang - 0.4), c[1] - 10 * Math.sin(ang - 0.4));
    ctx.lineTo(c[0] - 10 * Math.cos(ang + 0.4), c[1] - 10 * Math.sin(ang + 0.4));
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.fillStyle = C.mid;
    ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText(label, c[0] + 6, c[1] - 6);
  }

  // 조작
  const pre = presets((Object.keys(MODES) as Mode[]).map((k) => ({ key: k, label: MODES[k].label })), (k) => {
    mode = k; b = [...MODES[k].b]; delta = 0; pd.set(0); syncB(); update();
  }, mode);
  const fmt = (v: number) => (v >= 0 ? " " : "") + v.toFixed(1);
  const pbs = ["b₁", "b₂", "b₃"].map((label, i) => param({
    label, min: -6, max: 6, step: 0.5, value: b[i] ?? 0, fmt, aria: `벡터 b의 ${label} 성분`,
    onInput: (v) => { b[i] = v; update(); },
  }));
  const pd = param({
    label: "Δx", min: -2, max: 2, step: 0.1, value: 0, fmt, aria: "x를 최소제곱 해 x̂에서 옮기는 양",
    onInput: (v) => { delta = v; update(); },
  });
  const grid = el("div", "param-grid");
  pbs.forEach((p) => grid.append(p.row));
  grid.append(pd.row);
  const ro = readouts();
  pre.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(pre.row, grid, ro.row);

  function syncB() {
    pbs.forEach((p, i) => { p.row.style.display = i < dim() ? "" : "none"; if (i < dim()) p.set(b[i]); });
    pd.row.querySelector("label")!.textContent = dim() === 2 ? "Δx" : "Δx₂";
    pd.input.setAttribute("aria-label", dim() === 2 ? "x를 최소제곱 해 x̂에서 옮기는 양" : "x₂를 최소제곱 해 x̂₂에서 옮기는 양");
  }
  const f2 = (v: number) => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2);
  const vec = (v: Vec) => `(${v.map(f2).join(", ")})`;
  function update() {
    const A = cols();
    const { x, p, r, Atr } = lstsq(A, b);
    const xc = candidate(x);
    const rc = sub(b, Ax(A, xc));
    const Atrc = A.map((c) => dotv(c, rc));
    const moved = Math.abs(delta) > 1e-9;
    ro.set("x", `x̂ = ${x.length === 1 ? f2(x[0]) : vec(x)}`);
    ro.set("p", `p = ${vec(p)}`);
    ro.set("r", `r = ${vec(r)}`);
    ro.set("atr", moved ? `Aᵀ(b − Ax) = ${vec(Atrc)}` : `Aᵀr = ${vec(Atr)}`, moved ? "hot" : "ok");
    ro.set("n", moved ? `‖b − Ax‖ = ${norm(rc).toFixed(3)} > ‖r‖ = ${norm(r).toFixed(3)}` : `‖r‖ = ${norm(r).toFixed(3)} (최소)`, moved ? "hot" : "");
    draw();
  }

  // 끌기: ℝ²에서는 b를, ℝ³에서는 시점을
  const pos = (e: PointerEvent) => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] as [number, number]; };
    const nearB = (x: number, y: number) => { const q = toPx(b, scaleOf()); return Math.hypot(q[0] - x, q[1] - y) < 20; };
  f.canvas.addEventListener("pointerdown", (e) => {
    const [x, y] = pos(e);
    if (dim() === 2) { if (nearB(x, y)) drag = "b"; }
    else drag = "view";
    if (drag) { f.canvas.setPointerCapture(e.pointerId); last = [x, y]; }
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (!drag) { f.canvas.style.cursor = dim() === 3 ? "grab" : nearB(x, y) ? "grab" : "default"; return; }
    if (drag === "view") {
      az += (x - last[0]) * 0.01;
      el3 = Math.max(-1.2, Math.min(1.2, el3 + (y - last[1]) * 0.01));
      last = [x, y];
      draw();
      return;
    }
    const s = scaleOf();
    const snap = (v: number) => Math.max(-6, Math.min(6, Math.round(v * 2) / 2));
    b = [snap((x - f.width / 2) / s), snap((f.height / 2 - y) / s)];
    syncB();
    update();
  });
  const end = () => { if (drag) { drag = null; draw(); } };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  syncB();
  f.onResize(draw);
  update();
}
