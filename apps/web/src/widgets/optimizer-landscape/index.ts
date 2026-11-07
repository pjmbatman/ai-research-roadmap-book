// 손실 지형 위의 optimizer 궤적 (data/widgets.toml: optimizer-landscape).
// 2차원 손실 f(x, y)의 등고선 위에 GD · Momentum(heavy ball, β = 0.9) · Adam(β₁ = 0.9, β₂ = 0.999, ε = 1e−8, bias correction)의
// 궤적을 같은 학습률 η로 그린다. 이차형식에서는 GD의 안정 한계 2/L(L = Hessian의 최대 고유값)과 η를 비교한다.
// 조작: 지형 프리셋과 η 슬라이더(로그 눈금). 출발점은 그림에서 끌어 옮긴다.
import { frame, param, presets, readouts, el, C } from "../frame";
import { pageState } from "../../client/store";

type P2 = [number, number];
interface Land {
  label: string; f: (p: P2) => number; g: (p: P2) => P2; min: P2; hessMin: [number, number, number];
  box: [number, number, number, number]; start: P2; iters: number; lr: number; levels: [number, number];
}
const quad = (a: number, b: number, c: number): Pick<Land, "f" | "g" | "hessMin"> => ({
  f: ([x, y]) => 0.5 * (a * x * x + 2 * b * x * y + c * y * y),
  g: ([x, y]) => [a * x + b * y, b * x + c * y],
  hessMin: [a, b, c],
});
// 30° 돌린 이차형식: 고유값 1, 20
const th = Math.PI / 6, l1 = 1, l2 = 20;
const ra = l1 * Math.cos(th) ** 2 + l2 * Math.sin(th) ** 2, rb = (l1 - l2) * Math.cos(th) * Math.sin(th), rc = l1 * Math.sin(th) ** 2 + l2 * Math.cos(th) ** 2;
export const LANDS: Record<string, Land> = {
  quad: { label: "이차형식 κ = 10", ...quad(1, 0, 10), min: [0, 0], box: [-5, 5, -3, 3], start: [-4.5, 2.2], iters: 60, lr: 0.15, levels: [0.02, 60] },
  rot: { label: "회전된 이차형식 κ = 20", ...quad(ra, rb, rc), min: [0, 0], box: [-5, 5, -3, 3], start: [-4.5, -0.5], iters: 80, lr: 0.08, levels: [0.02, 90] },
  rosen: {
    label: "Rosenbrock",
    f: ([x, y]) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: ([x, y]) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    hessMin: [802, -400, 200], min: [1, 1], box: [-2, 2, -1, 3], start: [-1.5, 2.5], iters: 600, lr: 0.001, levels: [0.05, 2000],
  },
};

type Opt = "gd" | "mom" | "adam";
const OPTS: { key: Opt; name: string }[] = [{ key: "gd", name: "GD" }, { key: "mom", name: "Momentum" }, { key: "adam", name: "Adam" }];
/** 한 optimizer의 궤적. 발산(값이 1e6을 넘거나 NaN)하면 거기서 멈춘다. */
export function trajectory(L: Pick<Land, "f" | "g">, opt: Opt, lr: number, start: P2, iters: number) {
  let p: P2 = [...start];
  const path: P2[] = [[...p]];
  let v: P2 = [0, 0], m: P2 = [0, 0], s: P2 = [0, 0];
  const b1 = 0.9, b2 = 0.999, eps = 1e-8, beta = 0.9;
  let diverged = false;
  for (let t = 1; t <= iters; t++) {
    const g = L.g(p);
    if (opt === "gd") p = [p[0] - lr * g[0], p[1] - lr * g[1]];
    else if (opt === "mom") { v = [beta * v[0] + g[0], beta * v[1] + g[1]]; p = [p[0] - lr * v[0], p[1] - lr * v[1]]; }
    else {
      m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
      s = [b2 * s[0] + (1 - b2) * g[0] ** 2, b2 * s[1] + (1 - b2) * g[1] ** 2];
      const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
      p = [p[0] - (lr * (m[0] / c1)) / (Math.sqrt(s[0] / c2) + eps), p[1] - (lr * (m[1] / c1)) / (Math.sqrt(s[1] / c2) + eps)];
    }
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || Math.abs(p[0]) > 1e6 || Math.abs(p[1]) > 1e6) { diverged = true; break; }
    path.push([...p]);
  }
  return { path, diverged, final: diverged ? Infinity : L.f(p) };
}
export const lmax = ([a, b, c]: [number, number, number]) => (a + c) / 2 + Math.hypot((a - c) / 2, b);

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "손실 지형 위의 optimizer 궤적", "compare", root.clientWidth < 480 ? 0.85 : 0.62, "손실 등고선 위에 GD, Momentum, Adam의 궤적을 그린 그림");
  let key = attrs.preset in LANDS ? attrs.preset : "quad";
  let L = LANDS[key];
  let lr = L.lr;
  let start: P2 = [...L.start];
  let dragging = false;
  let contours: Path2D | null = null;
  let view = { cx: 0, cy: 0, s: 1 };
  const col = (o: Opt) => (o === "gd" ? C.ink : o === "mom" ? C.ink2 : C.edge);

  function fit() {
    const [x0, x1, y0, y1] = L.box, pad = 10;
    const s = Math.min((f.width - 2 * pad) / (x1 - x0), (f.height - 2 * pad) / (y1 - y0));
    view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s };
    contours = null;
  }
  const toPx = ([x, y]: P2): P2 => [f.width / 2 + (x - view.cx) * view.s, f.height / 2 - (y - view.cy) * view.s];
  const toW = (px: number, py: number): P2 => [view.cx + (px - f.width / 2) / view.s, view.cy - (py - f.height / 2) / view.s];

  /** 보이는 영역 전체의 등고선(marching squares, 로그 간격 수준) */
  function buildContours() {
    const path = new Path2D();
    const nx = Math.max(40, Math.min(140, Math.round(f.width / 5))), ny = Math.max(30, Math.round((nx * f.height) / f.width));
    const [ax, ay] = toW(0, 0), [bx, by] = toW(f.width, f.height);
    const val: number[][] = [];
    for (let j = 0; j <= ny; j++) { val.push([]); for (let i = 0; i <= nx; i++) val[j].push(Math.log(L.f([ax + ((bx - ax) * i) / nx, ay + ((by - ay) * j) / ny]) + 1e-12)); }
    const K = 14, lo = Math.log(L.levels[0]), hi = Math.log(L.levels[1]);
    const P = (i: number, j: number): P2 => toPx([ax + ((bx - ax) * i) / nx, ay + ((by - ay) * j) / ny]);
    for (let k = 0; k < K; k++) {
      const lv = lo + ((hi - lo) * k) / (K - 1);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const c = [val[j][i], val[j][i + 1], val[j + 1][i + 1], val[j + 1][i]];
        const corners: P2[] = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
        const pts: P2[] = [];
        for (let e = 0; e < 4; e++) {
          const a = c[e], b = c[(e + 1) % 4];
          if ((a < lv) !== (b < lv)) {
            const t = (lv - a) / (b - a), [i0, j0] = corners[e], [i1, j1] = corners[(e + 1) % 4];
            pts.push(P(i0 + (i1 - i0) * t, j0 + (j1 - j0) * t));
          }
        }
        for (let q = 0; q + 1 < pts.length; q += 2) { path.moveTo(...pts[q]); path.lineTo(...pts[q + 1]); }
      }
    }
    contours = path;
  }

  let runs = OPTS.map((o) => ({ ...o, ...trajectory(L, o.key, lr, start, L.iters) }));
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    if (!contours) buildContours();
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
    ctx.stroke(contours!);
    // 최소점
    const mn = toPx(L.min);
    ctx.strokeStyle = C.good; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(mn[0] - 6, mn[1] - 6); ctx.lineTo(mn[0] + 6, mn[1] + 6); ctx.moveTo(mn[0] + 6, mn[1] - 6); ctx.lineTo(mn[0] - 6, mn[1] + 6); ctx.stroke();
    // 궤적
    for (const r of [...runs].reverse()) {
      ctx.strokeStyle = col(r.key); ctx.fillStyle = col(r.key); ctx.lineWidth = 2; ctx.lineJoin = "round";
      // 화면 밖으로 멀리 나간 첫 점까지만 긋는다(발산하는 궤적이 그림을 덮지 않게)
      let last = r.path.findIndex((p) => { const q = toPx(p); return q[0] < -W / 4 || q[0] > 1.25 * W || q[1] < -H / 4 || q[1] > 1.25 * H; });
      const gone = last >= 0;
      if (!gone) last = r.path.length - 1;
      const pts = r.path.slice(0, last + 1).map(toPx);
      ctx.beginPath();
      pts.forEach((q, i) => (i ? ctx.lineTo(...q) : ctx.moveTo(...q)));
      ctx.stroke();
      if (r.path.length < 120) pts.forEach((q) => { ctx.beginPath(); ctx.arc(q[0], q[1], 2.2, 0, 7); ctx.fill(); });
      if (!gone) { const e = pts[pts.length - 1]; ctx.beginPath(); ctx.arc(e[0], e[1], 4.5, 0, 7); ctx.fill(); }
    }
    // 출발점(끌 수 있다)
    const s = toPx(start);
    ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(s[0], s[1], dragging ? 9 : 7, 0, 7); ctx.fill(); ctx.stroke();
    // 범례
    ctx.font = "600 11px JetBrains Mono, monospace";
    const lw = Math.max(...OPTS.map((o) => ctx.measureText(o.name).width)) + 34;
    // 범례는 오른쪽 아래(출발점 프리셋은 모두 왼쪽에 있다)
    const lx = W - lw - 8, ly = H - OPTS.length * 16 - 16;
    ctx.fillStyle = C.bg; ctx.globalAlpha = 0.85; ctx.fillRect(lx, ly, lw, OPTS.length * 16 + 8); ctx.globalAlpha = 1;
    OPTS.forEach((o, i) => {
      const y = ly + 14 + i * 16;
      ctx.strokeStyle = col(o.key); ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(lx + 6, y - 4); ctx.lineTo(lx + 22, y - 4); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.fillText(o.name, lx + 28, y);
    });
  }

  const pre = presets(Object.entries(LANDS).map(([k, l]) => ({ key: k, label: l.label })), (k) => {
    key = k; L = LANDS[k]; lr = L.lr; start = [...L.start]; pl.set(Math.log10(lr)); fit(); update();
  }, key);
  const pl = param({
    label: "η", min: -4, max: 0, step: 0.01, value: Math.log10(lr), fmt: (v) => (10 ** v).toPrecision(2),
    aria: "학습률 η(로그 눈금)", onInput: (v) => { lr = 10 ** v; update(); },
  });
  const grid = el("div", "param-grid");
  grid.append(pl.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);

  function update() {
    runs = OPTS.map((o) => ({ ...o, ...trajectory(L, o.key, lr, start, L.iters) }));
    const lim = 2 / lmax(L.hessMin);
    ro.set("lim", `GD 안정 한계 2/L = ${lim.toPrecision(2)}${key === "rosen" ? " (최소점)" : ""}`, lr > lim ? "hot" : "");
    ro.set("it", `반복 ${L.iters}회`);
    for (const r of runs) {
      const gap = r.final - L.f(L.min);
      const worse = r.diverged || r.final > L.f(start);
      ro.set(r.key, r.diverged ? `${r.name} 발산` : `${r.name} f = ${gap < 1e-12 ? "0" : gap.toExponential(1).replace(/-/g, "−")}${worse ? " · 커짐" : ""}`, worse ? "hot" : gap < 1e-4 ? "ok" : "");
    }
    pageState.set("optimizer", { landscape: key, lr, start: [...start] }, root);
    draw();
  }

  // 출발점 끌기
  const pos = (e: PointerEvent): P2 => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const near = (p: P2) => { const s = toPx(start); return Math.hypot(s[0] - p[0], s[1] - p[1]) < 18; };
  let raf = 0;
  f.canvas.addEventListener("pointerdown", (e) => { if (near(pos(e))) { dragging = true; f.canvas.setPointerCapture(e.pointerId); draw(); } });
  f.canvas.addEventListener("pointermove", (e) => {
    const p = pos(e);
    if (!dragging) { f.canvas.style.cursor = near(p) ? "grab" : "default"; return; }
    const w = toW(...p), [x0, x1, y0, y1] = L.box;
    start = [Math.max(x0, Math.min(x1, Math.round(w[0] * 20) / 20)), Math.max(y0, Math.min(y1, Math.round(w[1] * 20) / 20))];
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(update);
  });
  const end = () => { if (dragging) { dragging = false; draw(); } };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  f.onResize(() => { fit(); draw(); });
  fit();
  update();
}
