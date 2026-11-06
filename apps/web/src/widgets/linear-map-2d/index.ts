// 2차원 선형·아핀 사상 f(x) = A x + b (data/widgets.toml: linear-map-2d).
// 격자와 단위 정사각형이 어떻게 변형되는지, bias를 켜면 원점의 상 f(0)이 어디로 가는지 본다.
// 조작: 변형된 기저 벡터 끝(Ae1, Ae2)과 원점의 상 f(0)을 끌거나, 슬라이더·프리셋을 쓴다. '변형 보기'는 항등 사상에서 보간한다.
import { frame, param, presets, readouts, el, C, reducedMotion } from "../frame";
import { pageState } from "../../client/store";

type M = [number, number, number, number]; // a11 a12 a21 a22
const PRESETS: Record<string, { label: string; A: M; b: [number, number] }> = {
  identity: { label: "항등", A: [1, 0, 0, 1], b: [0, 0] },
  shear: { label: "전단(본문 예)", A: [1, 0.5, 0, 1], b: [0.5, 0.25] },
  rotate: { label: "회전 30°", A: [Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)], b: [0, 0] },
  scale: { label: "축별 확대", A: [1.5, 0, 0, 0.6], b: [0, 0] },
  singular: { label: "특이(rank 1)", A: [1, 0.5, 0.5, 0.25], b: [0, 0] },
};

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "2차원 선형·아핀 사상", "manipulate", 0.6, "변형된 격자와 단위 정사각형, 기저 벡터, 원점의 상을 그린 그림");
  const start = PRESETS[attrs.preset] ?? PRESETS.shear;
  let A: M = [...start.A];
  let b: [number, number] = [...start.b];
  let useBias = true;
  let t = 1; // 보간 진행도
  let drag: null | "e1" | "e2" | "b" = null;

  const scale = () => Math.min(f.width, f.height) / 6.4;
  const toPx = (x: number, y: number): [number, number] => [f.width / 2 + x * scale(), f.height / 2 - y * scale()];
  const toW = (px: number, py: number): [number, number] => [(px - f.width / 2) / scale(), (f.height / 2 - py) / scale()];
  const cur = (): { A: M; b: [number, number] } => {
    const bb: [number, number] = useBias ? b : [0, 0];
    return { A: [1 + (A[0] - 1) * t, A[1] * t, A[2] * t, 1 + (A[3] - 1) * t], b: [bb[0] * t, bb[1] * t] };
  };
  const apply = (m: { A: M; b: [number, number] }, x: number, y: number): [number, number] => [m.A[0] * x + m.A[1] * y + m.b[0], m.A[2] * x + m.A[3] * y + m.b[1]];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const m = cur();
    const R = 6;
    // 원래 격자
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(220,230,242,0.15)";
    for (let k = -R; k <= R; k++) {
      line(toPx(k, -R), toPx(k, R));
      line(toPx(-R, k), toPx(R, k));
    }
    // 변형된 격자
    ctx.strokeStyle = "rgba(220,230,242,0.26)";
    for (let k = -R; k <= R; k++) {
      line(toPx(...apply(m, k, -R)), toPx(...apply(m, k, R)));
      line(toPx(...apply(m, -R, k)), toPx(...apply(m, R, k)));
    }
    // 변형된 축
    ctx.strokeStyle = C.mid;
    line(toPx(...apply(m, -R, 0)), toPx(...apply(m, R, 0)));
    line(toPx(...apply(m, 0, -R)), toPx(...apply(m, 0, R)));
    // 단위 정사각형의 상
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => toPx(...apply(m, x, y)));
    ctx.beginPath();
    sq.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = "rgba(220,230,242,0.10)";
    ctx.fill();
    ctx.strokeStyle = "rgba(220,230,242,0.6)";
    ctx.stroke();
    // 기저 벡터의 상: 원점의 상에서 출발
    const o = toPx(...apply(m, 0, 0));
    arrow(o, toPx(...apply(m, 1, 0)), C.ink, "Ae₁");
    arrow(o, toPx(...apply(m, 0, 1)), C.ink2, "Ae₂");
    // 원래 원점과 원점의 상
    const o0 = toPx(0, 0);
    ctx.fillStyle = C.mid;
    dot(o0, 3);
    const moved = Math.hypot(m.b[0], m.b[1]) > 1e-9;
    if (moved) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.edge;
      line(o0, o);
      ctx.setLineDash([]);
    }
    ctx.fillStyle = moved ? C.edge : C.ink;
    dot(o, drag === "b" ? 8 : 6);
    ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText(moved ? "f(0) = b" : "f(0) = 0", o[0] + 10, o[1] + 16);
  }
  function line(a: [number, number], c: [number, number]) { f.ctx.beginPath(); f.ctx.moveTo(...a); f.ctx.lineTo(...c); f.ctx.stroke(); }
  function dot(p: [number, number], r: number) { f.ctx.beginPath(); f.ctx.arc(p[0], p[1], r, 0, 7); f.ctx.fill(); }
  function arrow(a: [number, number], c: [number, number], color: string, label: string) {
    const { ctx } = f;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.4;
    line(a, c);
    const ang = Math.atan2(c[1] - a[1], c[0] - a[0]);
    ctx.beginPath();
    ctx.moveTo(c[0], c[1]);
    ctx.lineTo(c[0] - 11 * Math.cos(ang - 0.4), c[1] - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(c[0] - 11 * Math.cos(ang + 0.4), c[1] - 11 * Math.sin(ang + 0.4));
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText(label, c[0] + 8, c[1] - 8);
  }

  // 조작 패널
  const fmt = (v: number) => (v >= 0 ? " " : "") + v.toFixed(2);
  const ps = ["a₁₁", "a₁₂", "a₂₁", "a₂₂"].map((label, i) =>
    param({ label, min: -2, max: 2, step: 0.05, value: A[i], fmt, aria: `행렬 A의 ${label} 성분`, onInput: (v) => { A[i] = v; pre.select(null); update(); } }),
  );
  const pb = ["b₁", "b₂"].map((label, i) =>
    param({ label, min: -2, max: 2, step: 0.05, value: b[i], fmt, aria: `bias ${label}`, onInput: (v) => { b[i] = v; pre.select(null); update(); } }),
  );
  const grid = el("div", "param-grid");
  [...ps, ...pb].forEach((p) => grid.append(p.row));
  const pre = presets(Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label })), (k) => {
    A = [...PRESETS[k].A]; b = [...PRESETS[k].b]; sync(); animate();
  }, attrs.preset in PRESETS ? attrs.preset : "shear");
  const tools = el("div", "wframe-row");
  const bias = el("label", "toggle");
  bias.innerHTML = `<input type="checkbox" checked /> bias b 켜기`;
  bias.querySelector("input")!.addEventListener("change", (e) => { useBias = (e.target as HTMLInputElement).checked; update(); });
  const play = el("button", "preset");
  play.type = "button";
  play.textContent = "▶ 항등에서 변형 보기";
  play.addEventListener("click", animate);
  tools.append(bias, play);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);

  function sync() { ps.forEach((p, i) => p.set(A[i])); pb.forEach((p, i) => p.set(b[i])); }
  function update(redraw = true) {
    const d = A[0] * A[3] - A[1] * A[2];
    const det = Math.abs(d) < 5e-3 ? 0 : d;
    const bb = useBias ? b : [0, 0];
    const linear = Math.hypot(bb[0], bb[1]) < 1e-9;
    ro.set("det", `det A = ${det.toFixed(2)}`, Math.abs(det) < 1e-9 ? "hot" : "");
    ro.set("f0", `f(0) = (${bb[0].toFixed(2)}, ${bb[1].toFixed(2)})`, linear ? "" : "hot");
    ro.set("lin", linear ? "선형사상" : "아핀 사상 · 선형 아님", linear ? "ok" : "hot");
    const z = (v: number) => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2);
    void z;
    pageState.set("A", [[A[0], A[1]], [A[2], A[3]]], root);
    pageState.set("b", [bb[0], bb[1]], root);
    if (redraw) draw();
  }
  function animate() {
    update(false);
    if (reducedMotion()) { t = 1; draw(); return; }
    const t0 = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / 600);
      t = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      draw();
      if (p < 1) requestAnimationFrame(step); else { t = 1; draw(); }
    };
    requestAnimationFrame(step);
  }

  // 끌기: 기저 벡터 끝과 원점의 상
  const handles = () => {
    const m = cur();
    return { b: toPx(...apply(m, 0, 0)), e1: toPx(...apply(m, 1, 0)), e2: toPx(...apply(m, 0, 1)) };
  };
  const pick = (x: number, y: number) => {
    const hs = handles();
    let best: typeof drag = null, d = 18;
    for (const k of ["b", "e1", "e2"] as const) {
      const dd = Math.hypot(hs[k][0] - x, hs[k][1] - y);
      if (dd < d) { d = dd; best = k; }
    }
    return best;
  };
  const pos = (e: PointerEvent) => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] as [number, number]; };
  f.canvas.addEventListener("pointerdown", (e) => {
    t = 1;
    drag = pick(...pos(e));
    if (drag) { f.canvas.setPointerCapture(e.pointerId); pre.select(null); }
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (!drag) { f.canvas.style.cursor = pick(x, y) ? "grab" : "default"; return; }
    const [wx, wy] = toW(x, y);
    const snap = (v: number) => Math.round(v * 20) / 20;
    if (drag === "b") { if (!useBias) return; b = [snap(wx), snap(wy)]; }
    else {
      const bb = useBias ? b : [0, 0];
      const col = drag === "e1" ? 0 : 1;
      A[col] = snap(wx - bb[0]);
      A[col + 2] = snap(wy - bb[1]);
    }
    A = A.map((v) => Math.max(-2, Math.min(2, v))) as M;
    sync();
    update();
  });
  const end = () => { drag = null; draw(); };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  f.onResize(draw);
  update();
}
