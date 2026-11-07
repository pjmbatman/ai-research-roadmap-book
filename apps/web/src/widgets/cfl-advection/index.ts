// CFL 조건 (data/widgets.toml: cfl-advection).
// 이류 방정식 u_t + c u_x = 0(c = 1 m/s, 길이 1 m 주기 경계, 격자 100칸)에서 가우시안 펄스를 한 바퀴(1 s) 돌린다(9.7장 셀과 같은 설정).
// upwind: u_j ← u_j − ν(u_j − u_{j−1}), 중심차분(FTCS): u_j ← u_j − (ν/2)(u_{j+1} − u_{j−1}), ν = cΔt/Δx.
// 왼쪽(좁은 화면에서는 위): 한 바퀴 뒤의 수치해와 정확해(처음 펄스). 오른쪽(아래): 격자 모드 e^{ijθ}의 한 step 증폭 인자 |G(θ)|.
import { frame, param, presets, readouts, el, C, alpha, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

const M = 100, DX = 1 / M;
const XS = Array.from({ length: M }, (_, j) => j * DX);
export const pulse = (x: number) => Math.exp(-200 * (x - 0.3) ** 2);
export type Scheme = "upwind" | "central";
const NAMES: Record<Scheme, string> = { upwind: "upwind", central: "중심차분" };

/** 한 바퀴(1 s) 동안 푼다. 그림에 쓸 중간 상태를 frames개 남긴다. */
export function advect(scheme: Scheme, nu: number, frames = 90) {
  const steps = Math.round(1 / (nu * DX));
  let u = Float64Array.from(XS, pulse);
  const snaps: { t: number; u: Float64Array }[] = [{ t: 0, u: u.slice() }];
  const every = Math.max(1, Math.round(steps / frames));
  for (let n = 1; n <= steps; n++) {
    const v = new Float64Array(M);
    for (let j = 0; j < M; j++) {
      const um = u[(j + M - 1) % M], up = u[(j + 1) % M];
      v[j] = scheme === "upwind" ? u[j] - nu * (u[j] - um) : u[j] - 0.5 * nu * (up - um);
    }
    u = v;
    if (n % every === 0 || n === steps) snaps.push({ t: n * nu * DX, u: u.slice() });
  }
  let maxU = -Infinity, err = 0;
  for (let j = 0; j < M; j++) { maxU = Math.max(maxU, u[j]); err = Math.max(err, Math.abs(u[j] - pulse(XS[j]))); }
  return { steps, snaps, maxU, err };
}
/** von Neumann 증폭 인자 |G(θ)|: upwind |G|² = 1 − 2ν(1 − ν)(1 − cos θ), 중심차분 |G|² = 1 + ν² sin²θ */
export const gain = (s: Scheme, nu: number, th: number) =>
  Math.sqrt(Math.max(0, s === "upwind" ? 1 - 2 * nu * (1 - nu) * (1 - Math.cos(th)) : 1 + nu * nu * Math.sin(th) ** 2));
export const maxGain = (s: Scheme, nu: number) => { let g = 0; for (let i = 0; i <= 180; i++) g = Math.max(g, gain(s, nu, (Math.PI * i) / 180)); return g; };

const sci = (v: number) => (v < 1e-12 ? "< 1e−12" : v.toExponential(2).replace("e+", "e").replace("e-", "e−"));
const big = (v: number) => (Math.abs(v) < 1000 ? v.toFixed(3) : v.toExponential(1).replace("e+", "e"));

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "CFL 조건", "manipulate", narrow ? 0.85 : 0.5, "주기 경계에서 한 바퀴 돈 가우시안 펄스의 수치해와 정확해, 그리고 격자 모드별 증폭 인자 |G(θ)|");
  let scheme: Scheme = "upwind", nu = 0.5;
  let sol = advect(scheme, nu), shown = sol.snaps.length - 1;
  const U0 = -0.8, U1 = 1.8, G1 = 1.8;

  function boxes() {
    const W = f.width, H = f.height;
    if (!narrow && W >= 480) {
      const gw = Math.min(240, W * 0.3);
      return { plot: { x: 34, y: 12, w: W - gw - 34 - 40, h: H - 36 }, gp: { x: W - gw - 12, y: 12, w: gw, h: H - 36 } };
    }
    const h1 = Math.round((H - 60) * 0.62);
    return { plot: { x: 34, y: 12, w: W - 46, h: h1 }, gp: { x: 34, y: 12 + h1 + 30, w: W - 46, h: H - h1 - 66 } };
  }

  function draw() {
    const { ctx, width: W } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, f.height);
    const { plot, gp } = boxes();
    const snap = sol.snaps[shown];
    const X = (x: number) => plot.x + x * plot.w;
    const Y = (u: number) => plot.y + ((U1 - u) / (U1 - U0)) * plot.h;
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
    ctx.textAlign = "right";
    for (const v of [0, 1]) { ctx.beginPath(); ctx.moveTo(plot.x, Y(v)); ctx.lineTo(plot.x + plot.w, Y(v)); ctx.stroke(); ctx.fillText(String(v), plot.x - 5, Y(v) + 4); }
    ctx.textAlign = "center";
    for (const v of [0, 0.5, 1]) ctx.fillText(`${v}`, X(v), plot.y + plot.h + 14);
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.w, plot.h); ctx.clip();
    // 정확해: 처음 펄스를 ct만큼 옮긴 것(주기 경계)
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i <= 300; i++) {
      const x = i / 300;
      let s = (((x - snap.t - 0.3) % 1) + 1) % 1; if (s > 0.5) s -= 1;
      const v = Math.exp(-200 * s * s);
      i ? ctx.lineTo(X(x), Y(v)) : ctx.moveTo(X(x), Y(v));
    }
    ctx.stroke();
    // 수치해
    ctx.strokeStyle = C.edge; ctx.fillStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let j = 0; j <= M; j++) { const v = Math.max(U0 - 1, Math.min(U1 + 1, snap.u[j % M])); j ? ctx.lineTo(X(j * DX), Y(v)) : ctx.moveTo(X(0), Y(v)); }
    ctx.stroke();
    if (plot.w / M > 4) for (let j = 0; j < M; j++) { ctx.beginPath(); ctx.arc(X(j * DX), Y(Math.max(U0 - 1, Math.min(U1 + 1, snap.u[j]))), 1.8, 0, 7); ctx.fill(); }
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    ctx.fillText("x (m)", X(0.78), plot.y + plot.h + 14);
    ctx.textAlign = "end";
    // 시각과 범례(발산한 해의 선 위에서도 읽히도록 바탕을 깐다)
    const tl = `t = ${snap.t.toFixed(2)} s`;
    ctx.textAlign = "start";
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(plot.x + 1, plot.y + 1, ctx.measureText(tl).width + 12, 20);
    ctx.fillStyle = C.mid; ctx.fillText(tl, plot.x + 6, plot.y + 14);
    ctx.textAlign = "end";
    const lx = plot.x + plot.w - 8;
    {
      const wl = Math.max(ctx.measureText("정확해").width, ctx.measureText(`수치해(${NAMES[scheme]})`).width) + 34;
      ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(lx - wl + 4, plot.y + 1, wl + 3, 38);
    }
    ctx.fillStyle = C.ink;
    ctx.fillText("정확해", lx, plot.y + 16); ctx.fillText(`수치해(${NAMES[scheme]})`, lx, plot.y + 32);
    const w1 = ctx.measureText("정확해").width, w2 = ctx.measureText(`수치해(${NAMES[scheme]})`).width;
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(lx - w1 - 24, plot.y + 12); ctx.lineTo(lx - w1 - 8, plot.y + 12); ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(lx - w2 - 24, plot.y + 28); ctx.lineTo(lx - w2 - 8, plot.y + 28); ctx.stroke();
    ctx.textAlign = "start";

    // 증폭 인자
    const GX = (th: number) => gp.x + (th / Math.PI) * gp.w;
    const GY = (g: number) => gp.y + ((G1 - Math.min(g, G1 + 0.2)) / G1) * gp.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(gp.x, gp.y, gp.w, gp.h);
    ctx.fillStyle = C.mid;
    ctx.fillText("|G(θ)|", gp.x + 6, gp.y + 14);
    {
      // 범례 겸 제목: 고른 방법은 코랄, 다른 방법은 흐린 색
      let lx = gp.x + 6 + ctx.measureText("|G(θ)| ").width;
      for (const s of ["upwind", "central"] as Scheme[]) {
        ctx.fillStyle = s === scheme ? C.edge : C.mid;
        ctx.fillText(NAMES[s], lx, gp.y + 14);
        lx += ctx.measureText(NAMES[s] + " ").width;
      }
      ctx.fillStyle = C.mid;
    }
    ctx.textAlign = "center";
    ctx.fillText("0", GX(0), gp.y + gp.h + 14); ctx.fillText("θ", GX(0.5 * Math.PI), gp.y + gp.h + 14); ctx.fillText("π", GX(Math.PI), gp.y + gp.h + 14);
    ctx.textAlign = "right";
    ctx.fillText("1", gp.x - 5, GY(1) + 4); ctx.fillText("0", gp.x - 5, GY(0) + 4);
    ctx.textAlign = "start";
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(gp.x, GY(1)); ctx.lineTo(gp.x + gp.w, GY(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.beginPath(); ctx.rect(gp.x, gp.y, gp.w, gp.h); ctx.clip();
    for (const s of (scheme === "upwind" ? ["central", "upwind"] : ["upwind", "central"]) as Scheme[]) {
      ctx.strokeStyle = s === scheme ? C.edge : C.grid2; ctx.lineWidth = s === scheme ? 2.4 : 1.6;
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) { const th = (Math.PI * i) / 120; i ? ctx.lineTo(GX(th), GY(gain(s, nu, th))) : ctx.moveTo(GX(th), GY(gain(s, nu, th))); }
      ctx.stroke();
    }
    ctx.restore();
  }

  let anim = 0;
  function play() {
    cancelAnimationFrame(anim);
    if (reducedMotion()) { shown = sol.snaps.length - 1; draw(); return; }
    const t0 = performance.now(), last = sol.snaps.length - 1;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / 2400);
      shown = Math.round(k * last);
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  function update() {
    cancelAnimationFrame(anim);
    sol = advect(scheme, nu);
    shown = sol.snaps.length - 1;
    const g = maxGain(scheme, nu);
    const unstable = g > 1 + 1e-12;
    ro.set("nu", `ν = ${nu.toFixed(2)} · step ${sol.steps}`);
    ro.set("g", `max |G| = ${g.toFixed(3)}${unstable ? " · 불안정" : ""}`, unstable ? "hot" : "ok");
    ro.set("max", `한 바퀴 뒤 최댓값 ${big(sol.maxU)}`);
    ro.set("err", `최대 오차 ${sci(sol.err)}`);
    pageState.set("cfl", { scheme, nu }, root);
    draw();
  }

  const sb = presets<Scheme>([{ key: "upwind", label: NAMES.upwind }, { key: "central", label: NAMES.central }], (k) => { scheme = k; update(); }, scheme);
  sb.row.classList.add("segmented");
  const pn = param({ label: "ν", min: 0.1, max: 1.3, step: 0.05, value: nu, fmt: (v) => v.toFixed(2), aria: "CFL 수 ν = cΔt/Δx", onInput: (v) => { nu = Math.round(v * 100) / 100; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pn.row);
  const tools = el("div", "wframe-row");
  const pb = el("button", "preset");
  pb.type = "button";
  pb.textContent = "▶ 한 바퀴 재생";
  pb.addEventListener("click", play);
  tools.append(pb);
  const ro = readouts();
  f.panel.append(sb.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
