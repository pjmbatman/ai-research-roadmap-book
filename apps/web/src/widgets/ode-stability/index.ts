// ODE solver의 step 크기와 안정성 (data/widgets.toml: ode-stability).
// 선형 문제 y′ = A y + g(t)를 explicit Euler · RK4 · implicit Euler로 같은 step 크기 h로 풀어 정확해와 비교한다.
// 오른쪽(좁은 화면에서는 아래) 그림은 복소평면의 안정 영역 |R(z)| ≤ 1과 지금의 z = hλ(λ는 A의 고유값)다.
// 조작: 문제 프리셋과 h 슬라이더(로그 눈금). h를 키우면 hλ가 안정 영역 밖으로 나가는 순간 그 방법이 발산한다.
import { frame, param, presets, readouts, el, C, alpha, rgbOf } from "../frame";
import { pageState } from "../../client/store";

type V = [number, number];
type Cx = [number, number];
interface Problem {
  label: string; A: [number, number, number, number]; g: (t: number) => V; y0: V; T: number;
  exact: (t: number) => number; eig: Cx[]; h: [number, number, number]; yr: [number, number];
}
const W4 = 4, Z = 0.1, WD = W4 * Math.sqrt(1 - Z * Z);
export const PROBLEMS: Record<string, Problem> = {
  decay: {
    label: "감쇠 y′ = −10y", A: [-10, 0, 0, 0], g: () => [0, 0], y0: [1, 0], T: 2,
    exact: (t) => Math.exp(-10 * t), eig: [[-10, 0]], h: [0.01, 0.4, 0.15], yr: [-1.3, 1.3],
  },
  osc: {
    label: "감쇠 진동자", A: [0, 1, -W4 * W4, -2 * Z * W4], g: () => [0, 0], y0: [1, 0], T: 8,
    exact: (t) => Math.exp(-Z * W4 * t) * (Math.cos(WD * t) + ((Z * W4) / WD) * Math.sin(WD * t)),
    eig: [[-Z * W4, WD], [-Z * W4, -WD]], h: [0.01, 0.6, 0.1], yr: [-1.6, 1.6],
  },
  stiff: {
    label: "stiff y′ = −50(y − cos t)", A: [-50, 0, 0, 0], g: (t) => [50 * Math.cos(t), 0], y0: [0, 0], T: 3,
    exact: (t) => (2500 / 2501) * Math.cos(t) + (50 / 2501) * Math.sin(t) - (2500 / 2501) * Math.exp(-50 * t),
    eig: [[-50, 0]], h: [0.005, 0.1, 0.05], yr: [-1.6, 1.6],
  },
};

const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cabs = (a: Cx) => Math.hypot(a[0], a[1]);
const sci = (v: number) => v.toExponential(1).replace(/-/g, "−");
/** 세 방법의 안정 함수: 시험 방정식 y′ = λy에서 y_{n+1} = R(hλ) y_n */
export const R = {
  euler: (z: Cx) => cabs([1 + z[0], z[1]]),
  rk4: (z: Cx) => { const z2 = cmul(z, z), z3 = cmul(z2, z), z4 = cmul(z3, z); return cabs([1 + z[0] + z2[0] / 2 + z3[0] / 6 + z4[0] / 24, z[1] + z2[1] / 2 + z3[1] / 6 + z4[1] / 24]); },
  implicit: (z: Cx) => 1 / cabs([1 - z[0], -z[1]]),
};
type Method = keyof typeof R;
const METHODS: { key: Method; name: string }[] = [
  { key: "euler", name: "explicit Euler" },
  { key: "rk4", name: "RK4" },
  { key: "implicit", name: "implicit Euler" },
];

/** 한 방법으로 [0, T]를 푼다. 발산하면 그 지점에서 멈춘다. */
export function solve(P: Problem, m: Method, h: number) {
  const [a, b, c, d] = P.A;
  const f = (t: number, y: V): V => { const g = P.g(t); return [a * y[0] + b * y[1] + g[0], c * y[0] + d * y[1] + g[1]]; };
  const N = Math.ceil(P.T / h - 1e-9);
  const ts = [0], ys = [P.y0[0]];
  let y: V = [...P.y0], diverged = false, maxErr = 0;
  for (let n = 0; n < N; n++) {
    const t = n * h;
    if (m === "euler") { const k = f(t, y); y = [y[0] + h * k[0], y[1] + h * k[1]]; }
    else if (m === "rk4") {
      const k1 = f(t, y);
      const k2 = f(t + h / 2, [y[0] + (h / 2) * k1[0], y[1] + (h / 2) * k1[1]]);
      const k3 = f(t + h / 2, [y[0] + (h / 2) * k2[0], y[1] + (h / 2) * k2[1]]);
      const k4 = f(t + h, [y[0] + h * k3[0], y[1] + h * k3[1]]);
      y = [y[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])];
    } else {
      // (I − hA) y_{n+1} = y_n + h g(t_{n+1})
      const g = P.g(t + h), r0 = y[0] + h * g[0], r1 = y[1] + h * g[1];
      const m11 = 1 - h * a, m12 = -h * b, m21 = -h * c, m22 = 1 - h * d, det = m11 * m22 - m12 * m21;
      y = [(m22 * r0 - m12 * r1) / det, (m11 * r1 - m21 * r0) / det];
    }
    if (!Number.isFinite(y[0]) || Math.abs(y[0]) > 1e6) { diverged = true; break; }
    ts.push(t + h); ys.push(y[0]);
    maxErr = Math.max(maxErr, Math.abs(y[0] - P.exact(t + h)));
  }
  return { ts, ys, diverged, maxErr, N };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = root.clientWidth < 480;
  const f = frame(root, "ODE solver의 step 크기와 안정성", "compare", narrow ? 1.08 : 0.62, "세 solver의 수치해와 정확해, 그리고 복소평면의 안정 영역과 hλ 위치");
  let key = attrs.preset in PROBLEMS ? attrs.preset : "stiff";
  let P = PROBLEMS[key];
  let h = P.h[2];
  const col = (m: Method | "exact") => (m === "euler" ? C.edge : m === "rk4" ? C.ink2 : m === "implicit" ? C.good : C.mid);

  function boxes() {
    const W = f.width, H = f.height;
    if (W >= 480) {
      const plot = { x: 34, y: 14, w: W * 0.62 - 34, h: H - 40 };
      const s = Math.min(W - plot.x - plot.w - 30, H - 96);
      return { plot, reg: { x: W - 12 - s, y: 14, s }, leg: { x: W - 12 - s, y: 14 + s + 22 } };
    }
    const plot = { x: 34, y: 12, w: W - 46, h: H * 0.55 - 30 };
    const s = Math.min(H - H * 0.55 - 10, W * 0.48);
    return { plot, reg: { x: 12, y: H - s - 6, s }, leg: { x: 12 + s + 16, y: H - s + 10 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const { plot, reg, leg } = boxes();
    const X = (t: number) => plot.x + (t / P.T) * plot.w;
    const Y = (y: number) => plot.y + ((P.yr[1] - y) / (P.yr[1] - P.yr[0])) * plot.h;
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
    ctx.beginPath(); ctx.moveTo(plot.x, Y(0)); ctx.lineTo(plot.x + plot.w, Y(0)); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.textAlign = "right";
    [P.yr[1], 0, P.yr[0]].forEach((v) => ctx.fillText(String(v).replace("-", "−"), plot.x - 5, Y(v) + 4));
    ctx.textAlign = "center";
    ctx.fillText("0", X(0), plot.y + plot.h + 15);
    ctx.fillText(`t = ${P.T}`, X(P.T) - 14, plot.y + plot.h + 15);
    ctx.textAlign = "start";
    ctx.save();
    ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.w, plot.h); ctx.clip();
    // 정확해
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 5; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i <= 400; i++) { const t = (i / 400) * P.T; i ? ctx.lineTo(X(t), Y(P.exact(t))) : ctx.moveTo(X(t), Y(P.exact(t))); }
    ctx.stroke();
    // 수치해
    const dots = (h / P.T) * plot.w > 7;
    for (const { key: m } of [...METHODS].reverse()) {
      const s = solve(P, m, h);
      // 그림 범위를 크게 벗어나면 그 점까지만 긋는다
      const span = P.yr[1] - P.yr[0];
      let last = s.ts.findIndex((_, i) => s.ys[i] > P.yr[1] + span || s.ys[i] < P.yr[0] - span);
      if (last < 0) last = s.ts.length - 1;
      ctx.strokeStyle = col(m); ctx.fillStyle = col(m); ctx.lineWidth = 2;
      if (m === "implicit") ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i <= last; i++) i ? ctx.lineTo(X(s.ts[i]), Y(s.ys[i])) : ctx.moveTo(X(s.ts[i]), Y(s.ys[i]));
      ctx.stroke();
      ctx.setLineDash([]);
      if (dots) for (let i = 0; i <= last; i++) { ctx.beginPath(); ctx.arc(X(s.ts[i]), Y(s.ys[i]), 2.6, 0, 7); ctx.fill(); }
    }
    ctx.restore();

    // 안정 영역: re ∈ [−5.2, 1.6], im ∈ [−3.4, 3.4]
    const re0 = -5.2, re1 = 1.6, im1 = 3.4, sc = reg.s / (re1 - re0);
    const ZX = (re: number) => reg.x + (re - re0) * sc, ZY = (im: number) => reg.y + reg.s / 2 - im * sc;
    const n = Math.max(40, Math.min(170, Math.round(reg.s)));
    const off = document.createElement("canvas"); off.width = off.height = n;
    const octx = off.getContext("2d")!;
    const img = octx.createImageData(n, n);
    const [r1, g1, b1] = rgbOf(C.ink2), [r2, g2, b2] = rgbOf(C.edge);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const z: Cx = [re0 + ((i + 0.5) / n) * (re1 - re0), im1 - ((j + 0.5) / n) * 2 * im1];
      const inRk = R.rk4(z) <= 1, inEu = R.euler(z) <= 1, k = 4 * (j * n + i);
      if (inEu) { img.data[k] = r2; img.data[k + 1] = g2; img.data[k + 2] = b2; img.data[k + 3] = 90; }
      else if (inRk) { img.data[k] = r1; img.data[k + 1] = g1; img.data[k + 2] = b1; img.data[k + 3] = 110; }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, reg.x, reg.y, reg.s, reg.s);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(reg.x, reg.y, reg.s, reg.s);
    ctx.beginPath(); ctx.moveTo(reg.x, ZY(0)); ctx.lineTo(reg.x + reg.s, ZY(0)); ctx.moveTo(ZX(0), reg.y); ctx.lineTo(ZX(0), reg.y + reg.s); ctx.stroke();
    // explicit Euler 경계(원 |1 + z| = 1)와 implicit Euler 경계(원 |1 − z| = 1, 바깥이 안정)
    ctx.save(); ctx.beginPath(); ctx.rect(reg.x, reg.y, reg.s, reg.s); ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = C.edge; ctx.beginPath(); ctx.arc(ZX(-1), ZY(0), sc, 0, 7); ctx.stroke();
    ctx.setLineDash([4, 3]); ctx.strokeStyle = C.good; ctx.beginPath(); ctx.arc(ZX(1), ZY(0), sc, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText("Re z", reg.x + reg.s - 30, ZY(0) - 4);
    ctx.fillText("Im z", ZX(0) + 4, reg.y + 11);
    // z = hλ
    P.eig.forEach((lam, i) => {
      const z: Cx = [h * lam[0], h * lam[1]];
      const out = z[0] < re0 || z[0] > re1 || Math.abs(z[1]) > im1;
      const px = Math.max(reg.x + 5, Math.min(reg.x + reg.s - 5, ZX(z[0]))), py = Math.max(reg.y + 5, Math.min(reg.y + reg.s - 5, ZY(z[1])));
      ctx.fillStyle = C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 7); ctx.fill(); ctx.stroke();
      if (i === 0) {
        ctx.font = "600 11px JetBrains Mono, monospace"; ctx.fillStyle = C.ink;
        ctx.fillText(out ? "← hλ" : "hλ", px + 8, py - 7);
      }
    });
    // 범례
    ctx.font = "11px JetBrains Mono, monospace";
    ([["exact", "정확해"], ...METHODS.map((m) => [m.key, m.name])] as [Method | "exact", string][]).forEach(([m, name], i) => {
      const y = leg.y + i * 16;
      ctx.strokeStyle = m === "exact" ? C.grid2 : col(m); ctx.lineWidth = m === "exact" ? 5 : 2.4;
      if (m === "implicit") ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(leg.x, y - 4); ctx.lineTo(leg.x + 16, y - 4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.ink; ctx.fillText(name, leg.x + 22, y);
    });
  }

  // 조작
  const pre = presets(Object.entries(PROBLEMS).map(([k, p]) => ({ key: k, label: p.label })), (k) => {
    key = k; P = PROBLEMS[k]; h = P.h[2];
    ph.input.min = String(Math.log10(P.h[0])); ph.input.max = String(Math.log10(P.h[1]));
    ph.set(Math.log10(h));
    update();
  }, key);
  const ph = param({
    label: "h", min: Math.log10(P.h[0]), max: Math.log10(P.h[1]), step: 0.01, value: Math.log10(h),
    fmt: (v) => (10 ** v).toPrecision(2), aria: "step 크기 h(로그 눈금)", onInput: (v) => { h = 10 ** v; update(); },
  });
  const grid = el("div", "param-grid");
  grid.append(ph.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);

  function update() {
    const z = P.eig.map((l) => [h * l[0], h * l[1]] as Cx);
    const fz = (v: number) => v.toFixed(2).replace("-", "−");
    ro.set("z", z.length === 2 ? `hλ = ${fz(z[0][0])} ± ${Math.abs(z[0][1]).toFixed(2)}i` : `hλ = ${fz(z[0][0])}`, "");
    ro.set("n", `step ${Math.ceil(P.T / h - 1e-9)}번`);
    for (const m of METHODS) {
      const amp = Math.max(...z.map((zz) => R[m.key](zz)));
      const s = solve(P, m.key, h);
      const bad = s.diverged || amp > 1 + 1e-12;
      ro.set(m.key, `${m.name} |R| = ${amp.toFixed(2)} · ${s.diverged ? "발산" : amp > 1 + 1e-12 ? `증폭 · 오차 ${sci(s.maxErr)}` : `오차 ${sci(s.maxErr)}`}`, bad ? "hot" : "ok");
    }
    pageState.set("odeStep", { problem: key, h }, root);
    draw();
  }

  f.onResize(draw);
  update();
}
