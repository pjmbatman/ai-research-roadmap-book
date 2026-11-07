// LQR과 제약 MPC (data/widgets.toml: lqr-mpc).
// 20.3장의 이중 적분기: 상태 (위치 [m], 속도 [m/s]), 입력 가속도 [m/s²], Δt = 0.1 s, x₀ = (1, 0), 60 step.
// 비용 Σ xᵀQx + R u², Q = diag(1, 0.1). R(입력 비용)을 바꾸면 Riccati 반복으로 K를 다시 구한다.
// 세 제어기: LQR(무제약) u = −Kx, LQR을 |u| ≤ 1에서 자른 것, 랜덤 슈팅 MPC(장과 같다: 후보 1000개, 지평 15,
// 입력은 [−1, 1] 균등 표집, 속도 |v| > 0.5 위반량에 1e3 벌점, 첫 입력만 적용). MPC 난수는 고정 시드다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

export const DT = 0.1, Q1 = 1, Q2 = 0.1, U_MAX = 1, V_MAX = 0.5, STEPS = 60;
type X2 = [number, number];
const step = (x: X2, u: number): X2 => [x[0] + DT * x[1] + 0.5 * DT * DT * u, x[1] + DT * u];

/** Riccati 반복: P ← Q + Aᵀ P (A − B K), K = (R + BᵀPB)⁻¹ BᵀPA */
export function lqrGain(R: number) {
  let p11 = Q1, p12 = 0, p22 = Q2, k1 = 0, k2 = 0, it = 0;
  const b1 = 0.5 * DT * DT, b2 = DT;
  for (it = 1; it <= 20000; it++) {
    // PB, BᵀPB, BᵀPA (A = [[1, Δt], [0, 1]])
    const pb1 = p11 * b1 + p12 * b2, pb2 = p12 * b1 + p22 * b2;
    const s = R + b1 * pb1 + b2 * pb2;
    k1 = pb1 / s; k2 = (DT * pb1 + pb2) / s;
    // A − BK
    const m11 = 1 - b1 * k1, m12 = DT - b1 * k2, m21 = -b2 * k1, m22 = 1 - b2 * k2;
    // P A_cl
    const a11 = p11 * m11 + p12 * m21, a12 = p11 * m12 + p12 * m22, a21 = p12 * m11 + p22 * m21, a22 = p12 * m12 + p22 * m22;
    // Q + Aᵀ (P A_cl), Aᵀ = [[1, 0], [Δt, 1]]
    const n11 = Q1 + a11, n12 = a12, n22 = Q2 + DT * a12 + a22;
    const d = Math.max(Math.abs(n11 - p11), Math.abs(n12 - p12), Math.abs(n22 - p22));
    p11 = n11; p12 = n12; p22 = n22;
    if (d < 1e-12) break;
  }
  return { K: [k1, k2] as X2, P: [p11, p12, p22], iters: it };
}

export type Ctrl = "lqr" | "clip" | "mpc";
export const CTRLS: { key: Ctrl; label: string }[] = [
  { key: "lqr", label: "LQR(무제약)" },
  { key: "clip", label: "LQR 자르기" },
  { key: "mpc", label: "MPC" },
];

function mpcPolicy(R: number, seed: number, N = 1000, Hh = 15) {
  const r = rng(seed);
  const U = new Float64Array(N * Hh);
  return (x: X2) => {
    for (let i = 0; i < U.length; i++) U[i] = (2 * r.u() - 1) * U_MAX;
    let best = Infinity, bi = 0;
    for (let n = 0; n < N; n++) {
      let p = x[0], v = x[1], J = 0;
      for (let h = 0; h < Hh; h++) {
        const u = U[n * Hh + h];
        J += Q1 * p * p + Q2 * v * v + R * u * u + 1e3 * Math.max(Math.abs(v) - V_MAX, 0);
        p = p + DT * v + 0.5 * DT * DT * u; v = v + DT * u;
      }
      if (J < best) { best = J; bi = n; }
    }
    return U[bi * Hh];
  };
}

/** 실제 시스템에서 60 step 굴린 궤적과 비용 */
export function rollout(ctrl: Ctrl, R: number) {
  const { K } = lqrGain(R);
  const pol = ctrl === "mpc" ? mpcPolicy(R, 7) : (x: X2) => {
    const u = -(K[0] * x[0] + K[1] * x[1]);
    return ctrl === "clip" ? Math.max(-U_MAX, Math.min(U_MAX, u)) : u;
  };
  let x: X2 = [1, 0], cost = 0;
  const xs: X2[] = [], us: number[] = [];
  for (let t = 0; t < STEPS; t++) {
    const u = pol(x);
    cost += Q1 * x[0] * x[0] + Q2 * x[1] * x[1] + R * u * u;
    xs.push(x); us.push(u);
    x = step(x, u);
  }
  xs.push(x);
  const maxU = Math.max(...us.map(Math.abs)), maxV = Math.max(...xs.slice(0, STEPS).map((s) => Math.abs(s[1])));
  return { xs, us, cost, maxU, maxV, finalX: Math.abs(xs[STEPS - 1][0]) };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "LQR과 제약 MPC", "compare", narrow ? 0.95 : 0.6, "세 제어기의 위치·속도·입력 궤적과 속도 한계 ±0.5 m/s, 입력 한계 ±1 m/s²");
  let ctrl: Ctrl = "clip", logR = -1;
  let runs: Record<Ctrl, ReturnType<typeof rollout>> = {} as Record<Ctrl, ReturnType<typeof rollout>>;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 44, PR = 10, top = 18, bottom = H - 22, gapY = 20;
    const bandH = (bottom - top - 2 * gapY) / 3;
    const X = (t: number) => PL + (t / STEPS) * (W - PL - PR);
    const strips: { name: string; lo: number; hi: number; lim?: number; get: (r: ReturnType<typeof rollout>) => number[] }[] = [
      { name: "위치 [m]", lo: -0.2, hi: 1.1, get: (r) => r.xs.slice(0, STEPS).map((s) => s[0]) },
      { name: "속도 [m/s]", lo: -1.0, hi: 0.3, lim: V_MAX, get: (r) => r.xs.slice(0, STEPS).map((s) => s[1]) },
      { name: "입력 [m/s²]", lo: -1.5, hi: 3.2, lim: U_MAX, get: (r) => r.us },
    ];
    // 속도·입력 축은 세 궤적과 한계가 모두 들어가게 정한다
    for (const i of [1, 2]) {
      const all = CTRLS.flatMap((c) => strips[i].get(runs[c.key]));
      const lim = strips[i].lim!;
      const lo = Math.min(-lim, ...all), hi = Math.max(lim, ...all), pad = 0.12 * (hi - lo);
      strips[i].lo = lo - pad; strips[i].hi = hi + pad;
    }
    strips.forEach((sp, i) => {
      const y0 = top + i * (bandH + gapY), y1 = y0 + bandH;
      const Y = (v: number) => y1 - ((v - sp.lo) / (sp.hi - sp.lo)) * (y1 - y0);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.strokeRect(PL, y0, W - PL - PR, bandH);
      ctx.beginPath(); ctx.moveTo(PL, Y(0)); ctx.lineTo(W - PR, Y(0)); ctx.stroke();
      ctx.fillStyle = C.mid;
      ctx.fillText("0", PL - 12, Y(0) + 4);
      ctx.fillText(sp.name, PL, y0 - 5);
      if (sp.lim !== undefined) {
        ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
        for (const s of [1, -1]) {
          const yl = Y(s * sp.lim);
          if (yl < y0 || yl > y1) continue;
          ctx.beginPath(); ctx.moveTo(PL, yl); ctx.lineTo(W - PR, yl); ctx.stroke();
          const t = `${s > 0 ? "" : "−"}${sp.lim}`;
          ctx.fillText(t, PL - 6 - ctx.measureText(t).width, yl + 4);
        }
        ctx.setLineDash([]);
      }
      ctx.save();
      ctx.beginPath(); ctx.rect(PL, y0, W - PL - PR, bandH); ctx.clip();
      // 고르지 않은 제어기는 흐리게, 고른 제어기는 코랄
      for (const c of [...CTRLS.filter((c) => c.key !== ctrl), CTRLS.find((c) => c.key === ctrl)!]) {
        const vals = sp.get(runs[c.key]);
        const sel = c.key === ctrl;
        ctx.strokeStyle = sel ? C.edge : alpha(C.ink2, 0.45);
        ctx.lineWidth = sel ? 2.2 : 1.3;
        ctx.beginPath();
        vals.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
        ctx.stroke();
        // 한계를 넘은 점 표시
        if (sel && sp.lim !== undefined && i === 1) {
          ctx.fillStyle = C.edge;
          vals.forEach((v, t) => { if (Math.abs(v) > sp.lim! + 1e-9) { ctx.beginPath(); ctx.arc(X(t), Y(v), 2.6, 0, 7); ctx.fill(); } });
        }
      }
      ctx.restore();
    });
    ctx.fillStyle = C.mid;
    for (const t of [0, 20, 40, 60]) { const s = `${(t * DT).toFixed(0)} s`; ctx.fillText(s, Math.min(X(t) - 8, W - PR - ctx.measureText(s).width), bottom + 15); }
  }

  let raf = 0;
  function update() {
    const R = 10 ** logR;
    runs = { lqr: rollout("lqr", R), clip: rollout("clip", R), mpc: rollout("mpc", R) };
    const r = runs[ctrl];
    ro.set("cost", `비용 ${r.cost.toFixed(3)} (LQR ${runs.lqr.cost.toFixed(3)})`, "");
    ro.set("v", `최대 |v| ${r.maxV.toFixed(2)} m/s`, r.maxV > V_MAX + 0.005 ? "hot" : "ok");
    ro.set("u", `최대 |u| ${r.maxU.toFixed(2)} m/s²`, r.maxU > U_MAX + 0.005 ? "hot" : "ok");
    ro.set("x", `최종 |x| ${r.finalX.toFixed(3)} m`);
    pageState.set("lqr-mpc", { R, ctrl, cost: r.cost, maxV: r.maxV }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets(CTRLS, (k) => { ctrl = k; update(); }, ctrl);
  pre.row.classList.add("segmented");
  const pr = param({ label: "R", min: -2, max: 0, step: 0.1, value: logR, fmt: (v) => { const r = 10 ** v; return r < 0.1 ? r.toFixed(3) : r < 1 ? r.toFixed(2) : r.toFixed(1); }, aria: "입력 비용 R (로그 눈금), Q = diag(1, 0.1)", onInput: (v) => { logR = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pr.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
