// 구면 위 보간 (data/widgets.toml: slerp-vs-lerp, 7.8장 "표현의 기하").
// 단위원 위의 두 점 p = (1, 0), q = (cos Ω, sin Ω) 사이를 세 방법으로 보간한다.
//   lerp  (1−t)p + tq             : 원 안쪽 현을 지나 노름이 줄어든다(t = 0.5에서 cos(Ω/2))
//   정규화 lerp  lerp / ‖lerp‖    : 원 위에 있지만 각이 고르게 늘지 않는다(가운데에서 빠르다)
//   slerp  본문 식                 : 원 위에서 각이 tΩ로 고르게 는다(측지선, 일정한 각속도)
// 조작: 방법 스위치, t 슬라이더, 사잇각 Ω 슬라이더(또는 그림에서 q 끌기).
import { frame, param, presets, readouts, el, C } from "../frame";

type V2 = [number, number];
export type Method = "lerp" | "nlerp" | "slerp";

export function interp(m: Method, om: number, t: number): V2 {
  const p: V2 = [1, 0], q: V2 = [Math.cos(om), Math.sin(om)];
  if (m === "slerp") {
    const a = Math.sin((1 - t) * om) / Math.sin(om), b = Math.sin(t * om) / Math.sin(om);
    return [a * p[0] + b * q[0], a * p[1] + b * q[1]];
  }
  const x: V2 = [(1 - t) * p[0] + t * q[0], (1 - t) * p[1] + t * q[1]];
  if (m === "lerp") return x;
  const n = Math.hypot(...x);
  return [x[0] / n, x[1] / n];
}
/** p에서 잰 각 θ(t) [rad] */
export const angleAt = (m: Method, om: number, t: number) => { const x = interp(m, om, t); return Math.atan2(x[1], x[0]); };
/** 각속도 dθ/dt [rad / 단위 t]. lerp와 정규화 lerp는 방향이 같아 sin Ω / ‖lerp‖² */
export function angSpeed(m: Method, om: number, t: number) {
  if (m === "slerp") return om;
  const x = interp("lerp", om, t);
  return Math.sin(om) / (x[0] * x[0] + x[1] * x[1]);
}

const METHODS: { key: Method; label: string }[] = [
  { key: "lerp", label: "lerp" },
  { key: "nlerp", label: "정규화 lerp" },
  { key: "slerp", label: "slerp" },
];
const deg = (r: number) => (r * 180) / Math.PI;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = root.clientWidth < 560;
  const f = frame(root, "구면 위 보간", "manipulate", narrow ? 1.05 : 0.52,
    "단위원 위 두 점 사이의 보간 경로와, t에 따른 각과 노름 그래프");
  let method: Method = (["lerp", "nlerp", "slerp"] as Method[]).includes(attrs.method as Method) ? (attrs.method as Method) : "lerp";
  let t = 0.5;
  let om = Math.PI / 2;
  let drag = false;

  function layout() {
    const W = f.width, H = f.height;
    if (W < 560) {
      const side = Math.min(W - 16, H * 0.46);
      const top = side + 14, gh = (H - top - 40) / 2;
      return { cx: W / 2, cy: 8 + side * 0.55, R: side * 0.4, plots: [{ x: 44, y: top + 12, w: W - 60, h: gh - 26 }, { x: 44, y: top + gh + 22, w: W - 60, h: gh - 26 }] };
    }
    const side = Math.min(H - 10, W * 0.5);
    const gx = side + 50, gw = W - gx - 18, gh = (H - 40) / 2;
    return { cx: side / 2 + 4, cy: H / 2 + side * 0.08, R: side * 0.4, plots: [{ x: gx, y: 22, w: gw, h: gh - 26 }, { x: gx, y: 22 + gh + 14, w: gw, h: gh - 26 }] };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const L = layout();
    const P = (v: V2): V2 => [L.cx + v[0] * L.R, L.cy - v[1] * L.R];
    // 축과 단위원
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    line(P([-1.2, 0]), P([1.2, 0])); line(P([0, -1.2]), P([0, 1.2]));
    ctx.strokeStyle = C.grid2;
    ctx.beginPath(); ctx.arc(L.cx, L.cy, L.R, 0, 2 * Math.PI); ctx.stroke();
    // 세 경로: lerp는 현, 나머지는 호. 고른 방법만 진하게
    const path = (m: Method) => {
      ctx.beginPath();
      for (let i = 0; i <= 60; i++) { const q = P(interp(m, om, i / 60)); if (i) ctx.lineTo(...q); else ctx.moveTo(...q); }
      ctx.stroke();
    };
    ctx.strokeStyle = C.edgeSoft; ctx.lineWidth = 2;
    path(method === "lerp" ? "slerp" : "lerp");
    ctx.strokeStyle = C.edge; ctx.lineWidth = 3;
    path(method);
    ctx.lineWidth = 1;
    // t = 0, 0.1, …, 1의 점: 간격이 고른지 본다
    ctx.fillStyle = C.edge;
    for (let i = 0; i <= 10; i++) dot(P(interp(method, om, i / 10)), 3);
    // p, q
    ctx.font = "700 13px JetBrains Mono, monospace";
    const p = P([1, 0]), q = P([Math.cos(om), Math.sin(om)]);
    ctx.fillStyle = C.ink;
    dot(p, 5); ctx.fillText("p", p[0] + 8, p[1] + 16);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(q[0], q[1], drag ? 9 : 7, 0, 7); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillText("q", q[0] + (Math.cos(om) >= 0 ? 10 : -20), q[1] - 8);
    // 현재 점 x(t)와 원점에서의 선
    const x = interp(method, om, t);
    ctx.strokeStyle = C.ink; ctx.setLineDash([4, 4]);
    line(P([0, 0]), P(x));
    ctx.setLineDash([]);
    ctx.fillStyle = C.edge;
    dot(P(x), 7);
    ctx.fillText("x(t)", P(x)[0] + 10, P(x)[1] - 10);

    // 그래프 두 개: θ(t)/Ω, ‖x(t)‖
    const plot = (g: { x: number; y: number; w: number; h: number }, title: string, fn: (m: Method, s: number) => number, ideal: (s: number) => number) => {
      const X = (s: number) => g.x + s * g.w;
      const Y = (v: number) => g.y + (1 - v) * g.h; // 값 범위 [0, 1]
      ctx.strokeStyle = C.faint;
      ctx.strokeRect(g.x, g.y, g.w, g.h);
      ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillText("0", g.x - 12, g.y + g.h + 4);
      ctx.fillText("1", g.x - 12, g.y + 4);
      ctx.fillText("t", g.x + g.w + 4, g.y + g.h + 4);
      ctx.fillStyle = C.ink; ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText(title, g.x, g.y - 6);
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
      ctx.beginPath();
      for (let i = 0; i <= 40; i++) { const s = i / 40; const pt: V2 = [X(s), Y(ideal(s))]; if (i) ctx.lineTo(...pt); else ctx.moveTo(...pt); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i <= 80; i++) { const s = i / 80; const pt: V2 = [X(s), Y(fn(method, s))]; if (i) ctx.lineTo(...pt); else ctx.moveTo(...pt); }
      ctx.stroke(); ctx.lineWidth = 1;
      ctx.strokeStyle = C.mid; line([X(t), g.y], [X(t), g.y + g.h]);
      ctx.fillStyle = C.edge; dot([X(t), Y(fn(method, t))], 5);
    };
    plot(L.plots[0], "θ(t) / Ω", (m, s) => angleAt(m, om, s) / om, (s) => s);
    plot(L.plots[1], "‖x(t)‖", (m, s) => Math.hypot(...interp(m, om, s)), () => 1);
  }
  function line(a: V2, c: V2) { f.ctx.beginPath(); f.ctx.moveTo(...a); f.ctx.lineTo(...c); f.ctx.stroke(); }
  function dot(p: V2, r: number) { f.ctx.beginPath(); f.ctx.arc(p[0], p[1], r, 0, 7); f.ctx.fill(); }

  // 조작
  const pre = presets(METHODS, (k) => { method = k; update(); }, method);
  pre.row.classList.add("segmented");
  const pt = param({ label: "t", min: 0, max: 1, step: 0.01, value: t, fmt: (v) => v.toFixed(2), aria: "보간 진행도 t", onInput: (v) => { t = v; update(); } });
  const po = param({ label: "Ω", min: 10, max: 170, step: 1, value: 90, fmt: (v) => `${v.toFixed(0)}°`, aria: "두 점 사이의 각 Ω(도)", onInput: (v) => { om = (v * Math.PI) / 180; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row, po.row);
  const ro = readouts();
  pre.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(pre.row, grid, ro.row);

  function update() {
    const x = interp(method, om, t);
    const n = Math.hypot(...x);
    const th = angleAt(method, om, t);
    const w = angSpeed(method, om, t) / om;
    ro.set("n", `‖x(t)‖ = ${n.toFixed(3)}`, Math.abs(n - 1) > 5e-4 ? "hot" : "ok");
    ro.set("th", `θ(t) = ${deg(th).toFixed(1)}° · tΩ = ${deg(t * om).toFixed(1)}°`, Math.abs(th - t * om) > 0.05 * Math.PI / 180 ? "hot" : "ok");
    ro.set("w", `dθ/dt = ${w.toFixed(3)} Ω`);
    ro.set("d", `측지 거리 Ω = ${om.toFixed(3)} · 직선 거리 = ${(2 * Math.sin(om / 2)).toFixed(3)}`);
    draw();
  }

  // q 끌기: 사잇각을 바꾼다
  const pos = (e: PointerEvent) => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] as V2; };
  const nearQ = (x: number, y: number) => { const L = layout(); return Math.hypot(L.cx + Math.cos(om) * L.R - x, L.cy - Math.sin(om) * L.R - y) < 20; };
  f.canvas.addEventListener("pointerdown", (e) => { const [x, y] = pos(e); if (nearQ(x, y)) { drag = true; f.canvas.setPointerCapture(e.pointerId); } });
  f.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (!drag) { f.canvas.style.cursor = nearQ(x, y) ? "grab" : "default"; return; }
    const L = layout();
    const a = Math.round(deg(Math.atan2(L.cy - y, x - L.cx)));
    const v = Math.max(10, Math.min(170, a < 0 ? (a < -90 ? 170 : 10) : a));
    om = (v * Math.PI) / 180;
    po.set(v);
    update();
  });
  const end = () => { if (drag) { drag = false; draw(); } };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  f.onResize(draw);
  update();
}
