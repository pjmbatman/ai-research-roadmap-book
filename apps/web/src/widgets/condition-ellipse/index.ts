// 조건수와 섭동 증폭 (data/widgets.toml: condition-ellipse).
// 2×2 행렬 A = U diag(1, 1/κ) Vᵀ(U는 35° 회전, V는 −20° 회전)로 Ax = b를 푼다. b = u₁(가장 큰 특잇값 방향)으로 두어 x = v₁, ‖x‖ = ‖b‖ = 1이다.
// 왼쪽: 크기 1인 섭동 Δb들이 이루는 원과 고른 Δb. 오른쪽: Δx = A⁻¹Δb들이 이루는 타원(반축 1/σᵢ, 방향 vᵢ)과 고른 Δx.
// 오른쪽 그림은 타원이 들어가게 축척을 바꾸고, 점선 원은 크기 1(왼쪽 원과 같은 크기)을 표시한다.
// readouts의 κ와 증폭률은 A의 성분에서 2×2 SVD를 다시 계산한 값이다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

type P2 = [number, number];
type M2 = [number, number, number, number];
const AU = (35 * Math.PI) / 180, AV = (-20 * Math.PI) / 180;
const rot = (t: number): M2 => [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];
const mul = (A: M2, B: M2): M2 => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
const tr = (A: M2): M2 => [A[0], A[2], A[1], A[3]];
const app = (A: M2, v: P2): P2 => [A[0] * v[0] + A[1] * v[1], A[2] * v[0] + A[3] * v[1]];

export function makeA(kappa: number): M2 {
  return mul(mul(rot(AU), [1, 0, 0, 1 / kappa]), tr(rot(AV)));
}
export function inv(A: M2): M2 {
  const d = A[0] * A[3] - A[1] * A[2];
  return [A[3] / d, -A[1] / d, -A[2] / d, A[0] / d];
}
/** 2×2 행렬의 특잇값 σ₁ ≥ σ₂ (AᵀA의 고윳값의 제곱근) */
export function svals(A: M2): [number, number] {
  const p = A[0] * A[0] + A[2] * A[2], q = A[0] * A[1] + A[2] * A[3], r = A[1] * A[1] + A[3] * A[3];
  const m = (p + r) / 2, h = Math.hypot((p - r) / 2, q);
  const det = Math.abs(A[0] * A[3] - A[1] * A[2]);
  const s1 = Math.sqrt(m + h);
  return [s1, det / s1]; // σ₂ = |det A| / σ₁ (작은 특잇값을 뺄셈 없이 구한다)
}
/** b = u₁, Δb = (cos φ, sin φ)에서 상대 오차 증폭률 (‖Δx‖/‖x‖)/(‖Δb‖/‖b‖) */
export function amplification(kappa: number, phi: number) {
  const A = makeA(kappa), Ai = inv(A);
  const b = app(rot(AU), [1, 0]);
  const x = app(Ai, b);
  const db: P2 = [Math.cos(phi), Math.sin(phi)];
  const dx = app(Ai, db);
  return (Math.hypot(...dx) / Math.hypot(...x)) / (Math.hypot(...db) / Math.hypot(...b));
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 480;
  const f = frame(root, "조건수와 섭동 증폭", "manipulate", narrow ? 1.6 : 0.5, "오른쪽 변의 섭동 Δb가 이루는 단위원과, 해의 섭동 Δx = A⁻¹Δb가 이루는 타원");
  let lk = Number(attrs.logk ?? 1); // log10 κ
  const u2Angle = AU + Math.PI / 2;
  let phi = u2Angle - 0.35; // Δb 방향
  let drag = false;

  const kappa = () => 10 ** lk;
  function geom() {
    const W = f.width, H = f.height;
    if (narrow) { const s = Math.min(W - 16, H / 2 - 16); return { L: [W / 2, 8 + s / 2] as P2, R: [W / 2, H / 2 + 8 + s / 2] as P2, r: s / 2 - 14 }; }
    const s = Math.min(W / 2 - 16, H - 16);
    return { L: [W / 4, H / 2] as P2, R: [(3 * W) / 4, H / 2] as P2, r: s / 2 - 14 };
  }

  function arrow(a: P2, b: P2, color: string, w: number) {
    const { ctx } = f;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.beginPath(); ctx.moveTo(...b);
    ctx.lineTo(b[0] - 11 * Math.cos(ang - 0.4), b[1] - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(b[0] - 11 * Math.cos(ang + 0.4), b[1] - 11 * Math.sin(ang + 0.4));
    ctx.fill();
  }
  function axis(c: P2, ang: number, len: number, label: string) {
    const { ctx } = f;
    const d: P2 = [Math.cos(ang), -Math.sin(ang)];
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(c[0] - len * d[0], c[1] - len * d[1]); ctx.lineTo(c[0] + len * d[0], c[1] + len * d[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.mid; ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText(label, c[0] + (len + 2) * d[0] - 6, c[1] + (len + 2) * d[1] + 4);
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const g = geom(), k = kappa(), A = makeA(k), Ai = inv(A);
    // 왼쪽: Δb 공간
    const sL = g.r * 0.78;
    axis(g.L, AU, g.r * 0.95, "u₁");
    axis(g.L, AU + Math.PI / 2, g.r * 0.95, "u₂");
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(g.L[0], g.L[1], sL, 0, 7); ctx.stroke();
    const db: P2 = [Math.cos(phi), Math.sin(phi)];
    const tipL: P2 = [g.L[0] + sL * db[0], g.L[1] - sL * db[1]];
    arrow(g.L, tipL, C.ink2, 2.6);
    ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(...tipL, drag ? 8 : 6, 0, 7); ctx.fill();
    // 오른쪽: Δx = A⁻¹Δb 공간. 축척은 타원의 긴 반축 1/σ₂ = κ가 들어가게.
    const sR = (g.r * 0.9) / k;
    axis(g.R, AV, g.r * 0.95, "v₁");
    axis(g.R, AV + Math.PI / 2, g.r * 0.95, "v₂");
    ctx.setLineDash([3, 3]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(g.R[0], g.R[1], Math.max(sR, 1.5), 0, 7); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i <= 160; i++) {
      const t = (2 * Math.PI * i) / 160, v = app(Ai, [Math.cos(t), Math.sin(t)]);
      const q: P2 = [g.R[0] + sR * v[0], g.R[1] - sR * v[1]];
      i ? ctx.lineTo(...q) : ctx.moveTo(...q);
    }
    ctx.fillStyle = alpha(C.edge, 0.12); ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.stroke();
    const dx = app(Ai, db);
    const tipR: P2 = [g.R[0] + sR * dx[0], g.R[1] - sR * dx[1]];
    if (Math.hypot(tipR[0] - g.R[0], tipR[1] - g.R[1]) > 12) arrow(g.R, tipR, C.edge, 2.6);
    else { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(...tipR, 4, 0, 7); ctx.fill(); }
    // 제목
    ctx.font = "600 12px Pretendard, sans-serif"; ctx.fillStyle = C.ink;
    const zoom = sL / sR, t1 = "Δb (‖Δb‖ = 1)", t2 = `Δx = A⁻¹Δb${zoom > 1.05 ? ` · 축척 1/${zoom >= 100 ? zoom.toFixed(0) : zoom.toFixed(1)}` : ""}`;
    ctx.fillText(t1, g.L[0] - g.r, g.L[1] - g.r + 2);
    ctx.fillText(t2, g.R[0] - g.r, g.R[1] - g.r + 2);
    if (!narrow) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(W / 2, 12); ctx.lineTo(W / 2, H - 12); ctx.stroke(); }
    else { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(12, H / 2); ctx.lineTo(W - 12, H / 2); ctx.stroke(); }
  }

  function update() {
    const k = kappa(), A = makeA(k), [s1, s2] = svals(A);
    const amp = amplification(k, phi);
    const Ai = inv(A), dx = app(Ai, [Math.cos(phi), Math.sin(phi)]);
    const big = (v: number) => (v >= 1000 ? v.toExponential(2).replace("e+", "e") : v.toFixed(v >= 100 ? 0 : 2));
    ro.set("k", `κ(A) = σ₁/σ₂ = ${big(s1 / s2)}`);
    ro.set("dx", `‖Δx‖/‖Δb‖ = ${big(Math.hypot(...dx))}`);
    ro.set("amp", `상대 오차 증폭 ${big(amp)} ≤ κ`, "hot");
    ro.set("dig", `잃는 자릿수 ≈ ${Math.log10(amp).toFixed(1)}`);
    draw();
  }

  const pk = param({ label: "κ", min: 0, max: 3, step: 0.05, value: lk, fmt: (v) => { const k = 10 ** v; return k >= 100 ? k.toFixed(0) : k.toFixed(1); }, aria: "조건수 κ(로그 눈금)", onInput: (v) => { lk = v; update(); } });
  const pd = param({ label: "Δb", min: 0, max: 180, step: 1, value: 0, fmt: (v) => `${v}°`, aria: "섭동 Δb의 방향(도, u₁에서 잰 각)", onInput: (v) => { phi = AU + (v * Math.PI) / 180; dirs.select(null); update(); } });
  const setPhi = (a: number) => { phi = a; let d = Math.round((((a - AU) * 180) / Math.PI) % 180); if (d < 0) d += 180; pd.set(d); };
  setPhi(phi);
  const dirs = presets([{ key: "u1", label: "Δb ∥ u₁" }, { key: "u2", label: "Δb ∥ u₂ (최악)" }], (kk) => { setPhi(kk === "u1" ? AU : u2Angle); update(); });
  const grid = el("div", "param-grid");
  grid.append(pk.row, pd.row);
  const ro = readouts();
  f.panel.append(dirs.row, grid, ro.row);

  // 끌기: 왼쪽 원 위의 Δb 끝
  const pos = (e: PointerEvent): P2 => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const onL = (q: P2) => { const g = geom(); return Math.hypot(q[0] - g.L[0], q[1] - g.L[1]) < g.r * 1.05; };
  const aim = (q: P2) => { const g = geom(); setPhi(Math.atan2(-(q[1] - g.L[1]), q[0] - g.L[0])); dirs.select(null); update(); };
  f.canvas.addEventListener("pointerdown", (e) => { const q = pos(e); if (!onL(q)) return; drag = true; f.canvas.setPointerCapture(e.pointerId); aim(q); });
  f.canvas.addEventListener("pointermove", (e) => { const q = pos(e); if (!drag) { f.canvas.style.cursor = onL(q) ? "grab" : "default"; return; } aim(q); });
  const end = () => { drag = false; draw(); };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);
  f.onResize(draw);
  update();
}
