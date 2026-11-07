// 격자 해상도와 셀 Péclet 수 (data/widgets.toml: grid-resolution-re). 26.1장 시뮬레이션 데이터 절의 예제 그대로다.
// 정상 이류-확산 U c_x = D c_xx, c(0) = 0, c(1) = 1을 N칸 격자로 푼다. 격자 Péclet 수 Pe_Δ = UΔx/D = Pe/N.
// 중심차분: −(1 + Pe_Δ/2)c_{j−1} + 2c_j − (1 − Pe_Δ/2)c_{j+1} = 0 (장의 코드와 같은 행렬).
// upwind:   −(1 + Pe_Δ)c_{j−1} + (2 + Pe_Δ)c_j − c_{j+1} = 0. 이것은 확산계수가 D(1 + Pe_Δ/2)인 중심차분과 같은 식이다(수치 확산).
// 정확해 c(x) = (e^{Pe x} − 1)/(e^{Pe} − 1). 같은 문제가 Burgers 방정식에서는 격자 레이놀즈 수 UΔx/ν로 나타난다.
// 왼쪽(좁은 화면에서는 위): 정확해와 수치해. 오른쪽(아래): 격자 수 N에 따른 격자점 최대 오차(두 방법), Pe_Δ = 2인 N.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export type Scheme = "central" | "upwind";
const NAMES: Record<Scheme, string> = { central: "중심차분", upwind: "upwind" };

/** 내부 격자점 N−1개의 해(양 끝 경계값 포함해 N+1개를 돌려준다). Thomas 알고리즘. */
export function solve(Pe: number, N: number, s: Scheme) {
  const P = Pe / N, n = N - 1;
  const lo = s === "central" ? -(1 + P / 2) : -(1 + P);
  const di = s === "central" ? 2 : 2 + P;
  const up = s === "central" ? -(1 - P / 2) : -1;
  const cp = new Float64Array(n), dp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const rhs = i === n - 1 ? -up * 1 : 0;   // 오른쪽 경계 c = 1을 우변으로
    const den = di - (i ? lo * cp[i - 1] : 0);
    cp[i] = up / den;
    dp[i] = (rhs - (i ? lo * dp[i - 1] : 0)) / den;
  }
  const c = new Float64Array(N + 1);
  c[N] = 1;
  for (let i = n - 1; i >= 0; i--) c[i + 1] = dp[i] - (i < n - 1 ? cp[i] * c[i + 2] : 0);
  return c;
}
/** 정확해(큰 Pe에서도 넘치지 않게) */
export const exact = (Pe: number, x: number) => (Math.exp(Pe * (x - 1)) - Math.exp(-Pe)) / (1 - Math.exp(-Pe));
export function errMax(Pe: number, N: number, s: Scheme) {
  const c = solve(Pe, N, s);
  let e = 0, mn = Infinity;
  for (let j = 0; j <= N; j++) { e = Math.max(e, Math.abs(c[j] - exact(Pe, j / N))); mn = Math.min(mn, c[j]); }
  return { err: e, min: mn };
}
const NS = Array.from({ length: 41 }, (_, i) => Math.round(4 * 100 ** (i / 40)));   // 4 … 400

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "격자 해상도와 셀 Péclet 수", "manipulate", narrow ? 1.0 : 0.5, "정상 이류-확산 방정식의 정확해와 격자 수치해, 격자 수에 따른 최대 오차");
  let Pe = 50, N = 10, scheme: Scheme = "central";

  function boxes() {
    const W = f.width, H = f.height;
    if (!narrow && W >= 400) {
      const gw = Math.min(230, W * 0.36);
      return { plot: { x: 38, y: 10, w: W - gw - 38 - 46, h: H - 36 }, gp: { x: W - gw - 10, y: 10, w: gw, h: H - 36 } };
    }
    const h1 = Math.round((H - 64) * 0.58);
    return { plot: { x: 38, y: 10, w: W - 48, h: h1 }, gp: { x: 38, y: h1 + 40, w: W - 48, h: H - h1 - 66 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { plot, gp } = boxes();
    const c = solve(Pe, N, scheme);
    let lo = -0.2, hi = 1.15;
    for (const v of c) { lo = Math.min(lo, v - 0.05); hi = Math.max(hi, v + 0.05); }
    lo = Math.max(lo, -3); hi = Math.min(hi, 4);
    const X = (x: number) => plot.x + x * plot.w;
    const Y = (v: number) => plot.y + ((hi - v) / (hi - lo)) * plot.h;
    ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
    ctx.strokeStyle = C.faint; ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 1]) { ctx.beginPath(); ctx.moveTo(plot.x, Y(v)); ctx.lineTo(plot.x + plot.w, Y(v)); ctx.stroke(); ctx.fillText(String(v), plot.x - 5, Y(v) + 4); }
    ctx.textAlign = "center";
    for (const v of [0, 0.5, 1]) ctx.fillText(String(v), X(v), plot.y + plot.h + 13);
    ctx.fillText("x", X(0.75), plot.y + plot.h + 13);
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.w, plot.h); ctx.clip();
    // 음수 영역
    if (Math.min(...c) < -1e-9) { ctx.fillStyle = alpha(C.edge, 0.08); ctx.fillRect(plot.x, Y(0), plot.w, Y(lo) - Y(0)); }
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i <= 400; i++) { const x = i / 400; i ? ctx.lineTo(X(x), Y(exact(Pe, x))) : ctx.moveTo(X(x), Y(exact(Pe, x))); }
    ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.fillStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); for (let j = 0; j <= N; j++) { const v = Math.max(lo - 1, Math.min(hi + 1, c[j])); j ? ctx.lineTo(X(j / N), Y(v)) : ctx.moveTo(X(0), Y(v)); } ctx.stroke();
    if (plot.w / N > 3.5) for (let j = 0; j <= N; j++) { ctx.beginPath(); ctx.arc(X(j / N), Y(Math.max(lo - 1, Math.min(hi + 1, c[j]))), 2.4, 0, 7); ctx.fill(); }
    ctx.restore();
    // 범례
    const t1 = "정확해", t2 = `수치해(${NAMES[scheme]})`;
    const wl = Math.max(ctx.measureText(t1).width, ctx.measureText(t2).width) + 30;
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(plot.x + 1, plot.y + 1, wl + 8, 38);
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(plot.x + 6, plot.y + 12); ctx.lineTo(plot.x + 22, plot.y + 12); ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(plot.x + 6, plot.y + 28); ctx.lineTo(plot.x + 22, plot.y + 28); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.fillText(t1, plot.x + 28, plot.y + 16); ctx.fillText(t2, plot.x + 28, plot.y + 32);
    // 오차 대 N (로그-로그)
    const E0 = -6, E1 = 1.8;
    const GX = (n: number) => gp.x + (Math.log10(n / 4) / 2) * gp.w;
    const GY = (e: number) => gp.y + ((E1 - Math.log10(Math.max(e, 1e-12))) / (E1 - E0)) * gp.h;
    ctx.lineWidth = 1; ctx.strokeStyle = C.faint; ctx.strokeRect(gp.x, gp.y, gp.w, gp.h);
    const n2 = Pe / 2;
    if (n2 > 4 && n2 < 400) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
      ctx.beginPath(); ctx.moveTo(GX(n2), gp.y); ctx.lineTo(GX(n2), gp.y + gp.h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText("Pe_Δ=2", Math.min(GX(n2) + 4, gp.x + gp.w - 46), gp.y + gp.h - 6);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(gp.x, gp.y, gp.w, gp.h); ctx.clip();
    for (const s of (scheme === "central" ? ["upwind", "central"] : ["central", "upwind"]) as Scheme[]) {
      ctx.strokeStyle = s === scheme ? C.edge : C.grid2; ctx.lineWidth = s === scheme ? 2.2 : 1.6;
      ctx.beginPath(); NS.forEach((n, i) => { const e = errMax(Pe, n, s).err; i ? ctx.lineTo(GX(n), GY(e)) : ctx.moveTo(GX(n), GY(e)); }); ctx.stroke();
    }
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(GX(N), GY(errMax(Pe, N, scheme).err), 4.5, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const n of [4, 40, 400]) ctx.fillText(String(n), GX(n), gp.y + gp.h + 13);
    ctx.fillText("N", GX(13), gp.y + gp.h + 13);
    ctx.textAlign = "right";
    for (const e of [1, 1e-3, 1e-6]) ctx.fillText(e === 1 ? "1" : e === 1e-3 ? "1e−3" : "1e−6", gp.x - 4, GY(e) + 4);
    ctx.textAlign = "start";
    let lx = gp.x + 6;
    ctx.fillStyle = C.mid; ctx.fillText("최대 오차", lx, gp.y + 14);
    lx += ctx.measureText("최대 오차 ").width;
    for (const s of ["central", "upwind"] as Scheme[]) { ctx.fillStyle = s === scheme ? C.edge : C.mid; ctx.fillText(NAMES[s], lx, gp.y + 14); lx += ctx.measureText(NAMES[s] + " ").width; }
  }

  function update() {
    const P = Pe / N, r = errMax(Pe, N, scheme);
    ro.set("pd", `Pe_Δ = ${P < 10 ? P.toFixed(2) : P.toFixed(1)}`, P > 2 && scheme === "central" ? "hot" : "");
    ro.set("min", r.min < -1e-9 ? `최솟값 ${r.min.toFixed(3)} · 음수 농도` : "음수 농도 없음", r.min < -1e-9 ? "hot" : "ok");
    ro.set("err", `최대 오차 ${r.err < 1e-3 ? r.err.toExponential(1).replace("e-", "e−") : r.err.toFixed(3)}`);
    ro.set("eff", scheme === "upwind" ? `유효 Pe = ${(Pe / (1 + P / 2)).toFixed(1)} (수치 확산)` : `Pe = ${Pe.toFixed(0)}`);
    draw();
  }

  const sb = presets<Scheme>([{ key: "central", label: NAMES.central }, { key: "upwind", label: NAMES.upwind }], (k) => { scheme = k; update(); }, scheme);
  sb.row.classList.add("segmented");
  const nice = (v: number) => Number((10 ** v).toPrecision(2));
  const pp = param({ label: "Pe", min: 0, max: 3, step: 0.05, value: Math.log10(Pe), fmt: (v) => String(nice(v)), aria: "전체 Péclet 수 Pe = UL/D(로그 눈금)", onInput: (v) => { Pe = nice(v); update(); } });
  const pn = param({ label: "N", min: 4, max: 200, step: 1, value: N, fmt: (v) => String(v), aria: "격자 칸 수 N", onInput: (v) => { N = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pp.row, pn.row);
  const ro = readouts();
  f.panel.append(sb.row, grid, ro.row);
  f.onResize(draw);
  update();
}
