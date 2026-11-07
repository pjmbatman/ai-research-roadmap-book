// SVD: 회전–스케일–회전 (data/widgets.toml: svd-explorer).
// 2×2 행렬 A = U Σ Vᵀ가 단위원을 어떻게 타원으로 바꾸는지 네 단계(x → Vᵀx → ΣVᵀx → UΣVᵀx)로 나눠 본다.
// 조작: 마지막 단계에서 Ae₁·Ae₂ 끝을 끌거나, 성분 슬라이더·프리셋으로 A를 바꾸고, 단계 버튼으로 한 단계씩 본다.
// V는 언제나 회전(det V = 1)으로 고르고, det A < 0이면 반사는 U에 둔다. 회전각은 (−90°, 90°] 범위에서 고른다.
import { frame, param, presets, readouts, el, C, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

type M = [number, number, number, number]; // a11 a12 a21 a22
export interface Svd { s1: number; s2: number; thV: number; thU: number; flip: 1 | -1 }

/** 2×2 실수 행렬의 SVD. A = R(thU)·diag(1, flip)·diag(s1, s2)·R(thV)ᵀ, s1 ≥ s2 ≥ 0. */
export function svd2(A: M): Svd {
  const [a, b, c, d] = A;
  const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d; // AᵀA = [[p, q], [q, r]]
  const mid = (p + r) / 2, rad = Math.hypot((p - r) / 2, q);
  const s1 = Math.sqrt(mid + rad);
  const det = a * d - b * c;
  const s2 = s1 > 1e-12 ? Math.abs(det) / s1 : 0;
  // AᵀA의 큰 고유벡터 방향: θ = ½·atan2(2q, p − r) ∈ (−90°, 90°]
  let thV = rad > 1e-12 ? 0.5 * Math.atan2(2 * q, p - r) : 0;
  if (thV <= -Math.PI / 2) thV += Math.PI;
  const v1: [number, number] = [Math.cos(thV), Math.sin(thV)];
  const flip: 1 | -1 = det < 0 ? -1 : 1;
  let thU = 0;
  if (s1 > 1e-12) {
    const u1 = [(a * v1[0] + b * v1[1]) / s1, (c * v1[0] + d * v1[1]) / s1];
    thU = Math.atan2(u1[1], u1[0]);
  }
  return { s1, s2, thV, thU, flip };
}
const rot = (t: number): M => [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];
const mul = (X: M, Y: M): M => [X[0] * Y[0] + X[1] * Y[2], X[0] * Y[1] + X[1] * Y[3], X[2] * Y[0] + X[3] * Y[2], X[2] * Y[1] + X[3] * Y[3]];
const diag = (x: number, y: number): M => [x, 0, 0, y];
/** 단계 k(0–3) 사이를 t로 보간한 사상: 0 = I, 1 = Vᵀ, 2 = ΣVᵀ, 3 = UΣVᵀ */
export function stageMap(S: Svd, k: number, t: number): M {
  const Vt = rot(-S.thV);
  if (k <= 0) return diag(1, 1);
  if (k === 1) return rot(-S.thV * t);
  const Sg = diag(1 + (S.s1 - 1) * t, 1 + (S.s2 - 1) * t);
  if (k === 2) return mul(Sg, Vt);
  const SV = mul(diag(S.s1, S.s2), Vt);
  return mul(mul(rot(S.thU * t), diag(1, 1 + (S.flip - 1) * t)), SV);
}

const PRESETS: Record<string, { label: string; A: M }> = {
  shear: { label: "전단", A: [1, 1, 0, 1] },
  sym: { label: "대칭", A: [1.5, 0.5, 0.5, 1] },
  rotate: { label: "회전", A: [Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)] },
  reflect: { label: "반사", A: [1.2, 0.4, 0.6, -0.8] },
  rank1: { label: "rank 1", A: [0.8, 0.4, 1.2, 0.6] },
};
const STAGES = [
  { key: "0", label: "① x", f: "x" },
  { key: "1", label: "② Vᵀx", f: "Vᵀx" },
  { key: "2", label: "③ ΣVᵀx", f: "ΣVᵀx" },
  { key: "3", label: "④ UΣVᵀx", f: "UΣVᵀx = Ax" },
] as const;
const VLAB = [["v₁", "v₂"], ["e₁", "e₂"], ["σ₁e₁", "σ₂e₂"], ["σ₁u₁", "σ₂u₂"]];

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "SVD: 회전–스케일–회전", "stepper", 0.62, "단위원이 Vᵀ 회전, Σ 스케일, U 회전을 차례로 거쳐 타원이 되는 그림과 특이벡터");
  const startKey = attrs.preset in PRESETS ? attrs.preset : "shear";
  let A: M = [...PRESETS[startKey].A];
  let S = svd2(A);
  let stage = 3, t = 1; // 지금 보이는 단계와 그 단계 안의 진행도
  let drag: null | 0 | 1 = null;
  let view = 1; // 화면에 들어가야 할 반지름(끄는 동안은 고정)
  let anim = 0;

  const scale = () => Math.min(f.width, f.height) / 2 / (view * 1.18);
  const toPx = (x: number, y: number): [number, number] => [f.width / 2 + x * scale(), f.height / 2 - y * scale()];
  const toW = (px: number, py: number): [number, number] => [(px - f.width / 2) / scale(), (f.height / 2 - py) / scale()];
  const ap = (m: M, x: number, y: number): [number, number] => [m[0] * x + m[1] * y, m[2] * x + m[3] * y];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const m = stageMap(S, stage, t);
    // 격자와 축
    const R = Math.ceil(view * 1.18 * Math.max(W, H) / Math.min(W, H)) + 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.faint;
    for (let k = -R; k <= R; k++) { line(toPx(k, -R), toPx(k, R)); line(toPx(-R, k), toPx(R, k)); }
    ctx.strokeStyle = C.grid2;
    line(toPx(-R, 0), toPx(R, 0));
    line(toPx(0, -R), toPx(0, R));
    // 원래 단위원(참조)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = C.mid;
    ctx.beginPath(); for (let i = 0; i <= 96; i++) { const a = (i / 96) * 2 * Math.PI; const p = toPx(Math.cos(a), Math.sin(a)); i ? ctx.lineTo(...p) : ctx.moveTo(...p); } ctx.stroke();
    ctx.setLineDash([]);
    // 지금 단계의 상: 원 → 타원
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) { const a = (i / 120) * 2 * Math.PI; const p = toPx(...ap(m, Math.cos(a), Math.sin(a))); i ? ctx.lineTo(...p) : ctx.moveTo(...p); }
    ctx.closePath();
    ctx.fillStyle = C.fill; ctx.fill();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2; ctx.stroke();
    // 원 위의 표지점(e₁의 상): 회전·반사가 보이도록
    ctx.fillStyle = C.ink2;
    dot(toPx(...ap(m, 1, 0)), 3.5);
    // 특이벡터의 상
    const o = toPx(0, 0);
    const v1 = [Math.cos(S.thV), Math.sin(S.thV)], v2 = [-Math.sin(S.thV), Math.cos(S.thV)];
    const lab = VLAB[t > 0.5 ? stage : Math.max(0, stage - 1)];
    arrow(o, toPx(...ap(m, v1[0], v1[1])), C.edge, lab[0]);
    arrow(o, toPx(...ap(m, v2[0], v2[1])), C.ink, lab[1]);
    // 마지막 단계에서는 A의 열(Ae₁, Ae₂)을 끌 수 있다
    if (stage === 3 && t === 1) {
      ([[A[0], A[2], "Ae₁"], [A[1], A[3], "Ae₂"]] as const).forEach(([x, y, s], i) => {
        const p = toPx(x, y);
        ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(p[0], p[1], drag === i ? 8 : 6, 0, 7); ctx.fill(); ctx.stroke();
        ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
        ctx.fillText(s, p[0] + 9, p[1] + 14);
      });
    }
    ctx.lineWidth = 1;
    // 지금 단계의 식
    ctx.fillStyle = C.ink;
    ctx.font = "600 13px JetBrains Mono, monospace";
    ctx.fillText(STAGES[t > 0.5 ? stage : Math.max(0, stage - 1)].f, 12, 20);
  }
  function line(a: [number, number], c: [number, number]) { f.ctx.beginPath(); f.ctx.moveTo(...a); f.ctx.lineTo(...c); f.ctx.stroke(); }
  function dot(p: [number, number], r: number) { f.ctx.beginPath(); f.ctx.arc(p[0], p[1], r, 0, 7); f.ctx.fill(); }
  function arrow(a: [number, number], c: [number, number], color: string, label: string) {
    const { ctx } = f;
    const len = Math.hypot(c[0] - a[0], c[1] - a[1]);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.4;
    line(a, c);
    if (len > 4) {
      const ang = Math.atan2(c[1] - a[1], c[0] - a[0]);
      const hl = Math.min(10, len * 0.5);
      ctx.beginPath();
      ctx.moveTo(c[0], c[1]);
      ctx.lineTo(c[0] - hl * Math.cos(ang - 0.4), c[1] - hl * Math.sin(ang - 0.4));
      ctx.lineTo(c[0] - hl * Math.cos(ang + 0.4), c[1] - hl * Math.sin(ang + 0.4));
      ctx.fill();
    }
    ctx.font = "600 12px JetBrains Mono, monospace";
    const ang = Math.atan2(c[1] - a[1], c[0] - a[0]);
    ctx.fillText(label, c[0] + 10 * Math.cos(ang) - 6, c[1] + 10 * Math.sin(ang) + 4);
  }

  // 조작 패널
  const stageBtns = presets(STAGES.map(({ key, label }) => ({ key, label })), (k) => go(Number(k)), "3");
  const fmt = (v: number) => (v >= 0 ? " " : "") + v.toFixed(2);
  const ps = ["a₁₁", "a₁₂", "a₂₁", "a₂₂"].map((label, i) =>
    param({ label, min: -2, max: 2, step: 0.05, value: A[i], fmt, aria: `행렬 A의 ${label} 성분`, onInput: (v) => { A[i] = v; pre.select(null); setA(true); } }),
  );
  const grid = el("div", "param-grid");
  ps.forEach((p) => grid.append(p.row));
  const pre = presets(Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label })), (k) => {
    A = [...PRESETS[k].A]; ps.forEach((p, i) => p.set(A[i])); setA(true); go(3, true);
  }, startKey);
  const tools = el("div", "wframe-row");
  const play = el("button", "preset");
  play.type = "button";
  play.textContent = "▶ 차례로 보기";
  play.addEventListener("click", () => go(3, true));
  tools.append(play);
  const ro = readouts();
  f.panel.append(stageBtns.row, pre.row, grid, tools, ro.row);

  function setA(fit: boolean) {
    S = svd2(A);
    if (fit) view = Math.max(1, S.s1);
    const tol = 1e-9 + 1e-6 * S.s1;
    const rank = S.s1 < tol ? 0 : S.s2 < Math.max(tol, 0.005) ? 1 : 2;
    const deg = (r: number) => `${(r * 180 / Math.PI).toFixed(0)}°`;
    ro.set("s1", `σ₁ = ${S.s1.toFixed(3)}`, "hot");
    ro.set("s2", `σ₂ = ${S.s2.toFixed(3)}`, rank < 2 ? "hot" : "");
    ro.set("rank", `rank ${rank}`, rank < 2 ? "hot" : "ok");
    ro.set("det", `|det A| = σ₁σ₂ = ${(S.s1 * S.s2).toFixed(3)}`);
    ro.set("rot", `V 회전 ${deg(S.thV)} · U 회전 ${deg(S.thU)}${S.flip < 0 ? " + 반사" : ""}`);
    pageState.set("A", [[A[0], A[1]], [A[2], A[3]]], root);
    pageState.set("singularValues", [S.s1, S.s2], root);
    draw();
  }

  /** 단계 target으로 간다. 앞으로 갈 때는 한 단계씩 움직임을 보인다. */
  function go(target: number, fromStart = false) {
    cancelAnimationFrame(anim);
    if (fromStart) { stage = 0; t = 1; }
    stageBtns.select(String(target) as "0" | "1" | "2" | "3");
    if (reducedMotion() || target <= stage) { stage = target; t = 1; draw(); return; }
    stage++; t = 0;
    let t0 = performance.now();
    const step = (now: number) => {
      const p = Math.max(0, Math.min(1, (now - t0) / 650));
      t = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      if (p >= 1) {
        t = 1;
        if (stage < target) { stage++; t = 0; t0 = now + 180; }
      }
      draw();
      if (stage < target || t < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  // 끌기: 마지막 단계에서 A의 열
  const pos = (e: PointerEvent) => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] as [number, number]; };
  const pick = (x: number, y: number): null | 0 | 1 => {
    if (stage !== 3 || t !== 1) return null;
    const hs = [toPx(A[0], A[2]), toPx(A[1], A[3])];
    const d = hs.map((h) => Math.hypot(h[0] - x, h[1] - y));
    const i = d[0] <= d[1] ? 0 : 1;
    return d[i] < 20 ? (i as 0 | 1) : null;
  };
  f.canvas.addEventListener("pointerdown", (e) => {
    drag = pick(...pos(e));
    if (drag !== null) { f.canvas.setPointerCapture(e.pointerId); pre.select(null); draw(); }
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (drag === null) { f.canvas.style.cursor = pick(x, y) !== null ? "grab" : "default"; return; }
    const [wx, wy] = toW(x, y);
    const snap = (v: number) => Math.max(-2, Math.min(2, Math.round(v * 20) / 20));
    A[drag] = snap(wx);
    A[drag + 2] = snap(wy);
    ps.forEach((p, i) => p.set(A[i]));
    setA(false);
  });
  const end = () => { if (drag === null) return; drag = null; setA(true); };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  f.onResize(draw);
  setA(true);
}
