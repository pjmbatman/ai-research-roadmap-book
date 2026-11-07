// 기울기와 Hessian 고유벡터 (data/widgets.toml: gradient-hessian-contour).
// 8.1장의 세 함수(이차 x² + 3y², 안장 x² − y², Rosenbrock (1 − x)² + 10(y − x²)²)의 등고선 위에서
// 끌 수 있는 점 p의 −∇f(코랄 화살표, 등고선에 수직)와 Hessian의 고유벡터(실선은 양의 곡률, 점선은 음의 곡률)를 그린다.
// gradient와 Hessian은 해석식으로 계산하고, 2×2 대칭행렬의 고윳값·고유벡터는 닫힌 식으로 구한다.
import { frame, presets, readouts, C } from "../frame";

type P2 = [number, number];
type H3 = [number, number, number]; // [hxx, hxy, hyy]
interface Fn {
  label: string; f: (p: P2) => number; g: (p: P2) => P2; H: (p: P2) => H3;
  box: [number, number, number, number]; start: P2; levels: { log: boolean; lo: number; hi: number; k: number };
}
const B = 10;
export const FNS: Record<string, Fn> = {
  quad: {
    label: "이차 x² + 3y²", f: ([x, y]) => x * x + 3 * y * y, g: ([x, y]) => [2 * x, 6 * y], H: () => [2, 0, 6],
    box: [-2.2, 2.2, -1.6, 1.6], start: [1, 1], levels: { log: false, lo: 0.25, hi: 12, k: 12 },
  },
  saddle: {
    label: "안장 x² − y²", f: ([x, y]) => x * x - y * y, g: ([x, y]) => [2 * x, -2 * y], H: () => [2, 0, -2],
    box: [-2, 2, -1.6, 1.6], start: [0.8, 0.5], levels: { log: false, lo: -3, hi: 3, k: 13 },
  },
  rosen: {
    label: "Rosenbrock", f: ([x, y]) => (1 - x) ** 2 + B * (y - x * x) ** 2,
    g: ([x, y]) => [-2 * (1 - x) - 4 * B * x * (y - x * x), 2 * B * (y - x * x)],
    H: ([x, y]) => [2 - 4 * B * y + 12 * B * x * x, -4 * B * x, 2 * B],
    box: [-2, 2, -1, 3], start: [0, 0.5], levels: { log: true, lo: 0.1, hi: 300, k: 13 },
  },
};

/** 2×2 대칭행렬의 고윳값(오름차순)과 단위 고유벡터 */
export function eig2([a, b, d]: H3): { lam: [number, number]; vec: [P2, P2] } {
  const m = (a + d) / 2, r = Math.hypot((a - d) / 2, b);
  const l1 = m - r, l2 = m + r;
  let v1: P2;
  if (Math.abs(b) > 1e-14) v1 = [b, l1 - a];
  else v1 = a <= d ? [1, 0] : [0, 1];
  const n = Math.hypot(...v1);
  v1 = [v1[0] / n, v1[1] / n];
  return { lam: [l1, l2], vec: [v1, [-v1[1], v1[0]]] };
}

export function classify(gn: number, lam: [number, number]) {
  const crit = gn < 1e-9;
  const zero = (l: number) => Math.abs(l) < 1e-9;
  if (zero(lam[0]) || zero(lam[1])) return crit ? "임계점 · 2차로 판정 불가" : "곡률 0인 방향 있음";
  const kind = lam[0] > 0 ? "양의 정부호" : lam[1] < 0 ? "음의 정부호" : "부정부호";
  if (!crit) return `임계점 아님 · H ${kind}`;
  return lam[0] > 0 ? "국소 최소점" : lam[1] < 0 ? "국소 최대점" : "안장점";
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 480;
  const f = frame(root, "기울기와 Hessian 고유벡터", "manipulate", narrow ? 0.9 : 0.6, "함수의 등고선 위에 끌 수 있는 점, 그 점의 음의 gradient 화살표와 Hessian 고유벡터를 그린 그림");
  let key = attrs.preset in FNS ? attrs.preset : "rosen";
  let F = FNS[key];
  let p: P2 = [...F.start];
  let dragging = false;
  let contours: Path2D | null = null;
  let view = { cx: 0, cy: 0, s: 1 };

  function fit() {
    const [x0, x1, y0, y1] = F.box, pad = 10;
    view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s: Math.min((f.width - 2 * pad) / (x1 - x0), (f.height - 2 * pad) / (y1 - y0)) };
    contours = null;
  }
  const toPx = ([x, y]: P2): P2 => [f.width / 2 + (x - view.cx) * view.s, f.height / 2 - (y - view.cy) * view.s];
  const toW = (px: number, py: number): P2 => [view.cx + (px - f.width / 2) / view.s, view.cy - (py - f.height / 2) / view.s];

  /** 보이는 영역의 등고선(marching squares) */
  function buildContours() {
    const path = new Path2D();
    const nx = Math.max(40, Math.min(150, Math.round(f.width / 5))), ny = Math.max(30, Math.round((nx * f.height) / f.width));
    const [ax, ay] = toW(0, 0), [bx, by] = toW(f.width, f.height);
    const L = F.levels, tr = (v: number) => (L.log ? Math.log(Math.max(v, 1e-12)) : v);
    const val: number[][] = [];
    for (let j = 0; j <= ny; j++) { val.push([]); for (let i = 0; i <= nx; i++) val[j].push(tr(F.f([ax + ((bx - ax) * i) / nx, ay + ((by - ay) * j) / ny]))); }
    const lo = tr(L.lo), hi = tr(L.hi);
    const P = (i: number, j: number): P2 => toPx([ax + ((bx - ax) * i) / nx, ay + ((by - ay) * j) / ny]);
    for (let k = 0; k < L.k; k++) {
      const lv = lo + ((hi - lo) * k) / (L.k - 1);
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

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    if (!contours) buildContours();
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
    ctx.stroke(contours!);
    // 축
    ctx.strokeStyle = C.faint;
    const o = toPx([0, 0]);
    ctx.beginPath(); ctx.moveTo(0, o[1]); ctx.lineTo(W, o[1]); ctx.moveTo(o[0], 0); ctx.lineTo(o[0], H); ctx.stroke();
    const pp = toPx(p);
    // Hessian 고유벡터: 양의 곡률은 실선, 음의 곡률은 점선. 길이는 고정, 굵기로 큰 곡률을 강조한다.
    const { lam, vec } = eig2(F.H(p));
    const amax = Math.max(Math.abs(lam[0]), Math.abs(lam[1]), 1e-12);
    ctx.font = "600 12px JetBrains Mono, monospace";
    for (let i = 0; i < 2; i++) {
      const v = vec[i], L = 46;
      const a: P2 = [pp[0] - L * v[0], pp[1] + L * v[1]], b: P2 = [pp[0] + L * v[0], pp[1] - L * v[1]];
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2 + 2 * (Math.abs(lam[i]) / amax);
      ctx.setLineDash(lam[i] < 0 ? [6, 4] : []);
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      ctx.setLineDash([]);
      const s = `λ${i ? "₂" : "₁"} = ${lam[i].toFixed(lam[i] !== 0 && Math.abs(lam[i]) < 10 ? 2 : 1)}`;
      const tx = Math.max(4, Math.min(W - ctx.measureText(s).width - 4, b[0] + (b[0] >= pp[0] ? 6 : -ctx.measureText(s).width - 6)));
      const ty = Math.max(14, Math.min(H - 6, b[1] + (b[1] >= pp[1] ? 14 : -6)));
      ctx.fillStyle = C.ink; ctx.fillText(s, tx, ty);
    }
    // −∇f: 길이는 |∇f|의 로그에 따라 18–90px
    const g = F.g(p), gn = Math.hypot(...g);
    if (gn > 1e-9) {
      const len = 18 + Math.min(72, 18 * Math.log10(1 + gn * 10));
      const tip: P2 = [pp[0] - (len * g[0]) / gn, pp[1] + (len * g[1]) / gn];
      arrow(pp, tip, C.edge, 2.6);
      ctx.fillStyle = C.edge;
      ctx.fillText("−∇f", Math.max(4, Math.min(W - 30, tip[0] - (10 * g[0]) / gn - 12)), Math.max(14, Math.min(H - 6, tip[1] + (14 * g[1]) / gn + 4)));
    }
    ctx.fillStyle = dragging ? C.edge : C.ink;
    ctx.beginPath(); ctx.arc(pp[0], pp[1], dragging ? 8 : 6.5, 0, 7); ctx.fill();
    ctx.strokeStyle = C.bg; ctx.lineWidth = 2; ctx.stroke();
  }

  function update() {
    const v = F.f(p), g = F.g(p), gn = Math.hypot(...g);
    const { lam } = eig2(F.H(p));
    const n = (x: number) => (Math.abs(x) < 5e-13 ? 0 : x);
    const fx = (x: number) => (Math.abs(x) >= 100 ? n(x).toFixed(0) : n(x).toFixed(2));
    ro.set("p", `p = (${fx(p[0])}, ${fx(p[1])})`);
    ro.set("f", `f = ${fx(v)}`);
    ro.set("g", `∇f = (${fx(g[0])}, ${fx(g[1])})`, gn < 1e-9 ? "ok" : "");
    ro.set("lam", `λ = ${fx(lam[0])}, ${fx(lam[1])}`);
    ro.set("cls", classify(gn, lam), "hot");
    ro.set("kappa", lam[0] > 0 ? `λ₂/λ₁ = ${(lam[1] / lam[0]).toFixed(1)}` : "λ₂/λ₁ 정의 안 됨");
    draw();
  }

  const pre = presets(Object.entries(FNS).map(([k, v]) => ({ key: k, label: narrow ? v.label.split(" ")[0] : v.label })), (k) => {
    key = k; F = FNS[k]; p = [...F.start]; fit(); update();
  }, key);
  const ro = readouts();
  f.panel.append(pre.row, ro.row);

  // 끌기: 점 p. 0.05 격자에 붙여서 최소점·안장점에 정확히 놓을 수 있게 한다.
  const pos = (e: PointerEvent): P2 => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const near = (q: P2) => { const pp = toPx(p); return Math.hypot(pp[0] - q[0], pp[1] - q[1]) < 22; };
  const moveTo = (q: P2) => {
    const [x0, x1, y0, y1] = F.box, w = toW(...q), snap = (v: number) => Math.round(v * 20) / 20;
    p = [Math.max(x0, Math.min(x1, snap(w[0]))), Math.max(y0, Math.min(y1, snap(w[1])))];
    update();
  };
  f.canvas.addEventListener("pointerdown", (e) => { dragging = true; f.canvas.setPointerCapture(e.pointerId); moveTo(pos(e)); });
  f.canvas.addEventListener("pointermove", (e) => {
    if (!dragging) { f.canvas.style.cursor = near(pos(e)) ? "grab" : "crosshair"; return; }
    moveTo(pos(e));
  });
  const end = () => { dragging = false; draw(); };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);
  // 키보드: canvas에 초점을 두고 화살표 키로 0.05씩 옮긴다.
  f.canvas.tabIndex = 0;
  f.canvas.addEventListener("keydown", (e) => {
    const d: Record<string, P2> = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] };
    if (!(e.key in d)) return;
    e.preventDefault();
    const [x0, x1, y0, y1] = F.box, r2 = (v: number) => Math.round(v * 20) / 20;
    p = [Math.max(x0, Math.min(x1, r2(p[0] + d[e.key][0]))), Math.max(y0, Math.min(y1, r2(p[1] + d[e.key][1])))];
    update();
  });

  f.onResize(() => { fit(); draw(); });
  fit();
  update();
}
