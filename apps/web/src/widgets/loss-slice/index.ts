// 손실 단면과 보간 경로 (data/widgets.toml: loss-slice).
// 합성(가상) 손실 50차원: θ = (u, v, z₁…z₄₈),
//   L(θ) = (u² + v² − 1)² + v²(0.175 − 0.125 tanh 3v) + ½ Σ λ_k z_k²,  λ = (40, 12, 5, 1 … 0.01 로그 간격 45개).
// 두 해 θ_a = (−1, 0, 0…), θ_b = (1, 0, 0…)는 u ↦ −u 대칭으로 손실이 같다(뉴런 순서를 바꾼 해처럼).
// 경로 단면: 직선 (1 − α)θ_a + αθ_b, (u, v) 평면 골짜기 바닥을 따라가는 곡선. 방향 단면: θ_a + t·d(|d| = 1), d는 무작위 또는 Hessian 최대 고유벡터.
// 왼쪽(위): (u, v) 평면의 손실 등고선과 경로. 오른쪽(아래): 1D 손실 단면.
import { frame, param, presets, readouts, el, C, rng, rgbOf } from "../frame";
import { pageState } from "../../client/store";

const D = 50;
export const LAM = [40, 12, 5, ...Array.from({ length: 45 }, (_, i) => 10 ** (-2 * i / 44))];
export const Lp = (u: number, v: number) => (u * u + v * v - 1) ** 2 + v * v * (0.175 - 0.125 * Math.tanh(3 * v));
export function loss(th: ArrayLike<number>) {
  let s = Lp(th[0], th[1]);
  for (let k = 0; k < 48; k++) s += 0.5 * LAM[k] * th[k + 2] ** 2;
  return s;
}
const A = (() => { const a = new Float64Array(D); a[0] = -1; return a; })();
/** θ_a에서의 Hessian은 대각: ∂²L/∂u² = 8, ∂²L/∂v² = 0.35, ∂²L/∂z_k² = λ_k */
export const HDIAG = [8, 0.35, ...LAM];
export const curvature = (d: ArrayLike<number>) => { let s = 0; for (let i = 0; i < D; i++) s += HDIAG[i] * d[i] * d[i]; return s; };

/** (u, v) 평면의 위쪽 골짜기 바닥: 각 φ에서 r을 황금분할 탐색으로 최소화하고, 호 길이로 다시 매개화한다. */
export const VALLEY = (() => {
  const K = 400, pts: [number, number][] = [];
  for (let i = 0; i <= K; i++) {
    const phi = Math.PI * (1 - i / K);
    const g = (r: number) => Lp(r * Math.cos(phi), r * Math.sin(phi));
    let a = 0.5, b = 1.5;
    const q = (Math.sqrt(5) - 1) / 2;
    for (let it = 0; it < 60; it++) { const c = b - q * (b - a), d = a + q * (b - a); if (g(c) < g(d)) b = d; else a = c; }
    const r = (a + b) / 2;
    pts.push([r * Math.cos(phi), r * Math.sin(phi)]);
  }
  const s = [0];
  for (let i = 1; i <= K; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return { pts, s, len: s[K] };
})();
function onValley(alpha: number): [number, number] {
  const target = alpha * VALLEY.len, { s, pts } = VALLEY;
  let i = 1;
  while (i < s.length - 1 && s[i] < target) i++;
  const w = (target - s[i - 1]) / (s[i] - s[i - 1] || 1);
  return [pts[i - 1][0] + w * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + w * (pts[i][1] - pts[i - 1][1])];
}

export type Mode = "line" | "curve" | "rand" | "top";
const MODES: { key: Mode; label: string }[] = [
  { key: "line", label: "직선 경로" }, { key: "curve", label: "곡선 경로" }, { key: "rand", label: "무작위 방향" }, { key: "top", label: "최대 곡률 방향" },
];
export function randomDir(seed: number) {
  const r = rng(seed), d = new Float64Array(D);
  let n = 0;
  for (let i = 0; i < D; i++) { d[i] = r.normal(); n += d[i] * d[i]; }
  for (let i = 0; i < D; i++) d[i] /= Math.sqrt(n);
  return d;
}
const TOP = (() => { const d = new Float64Array(D); d[2] = 1; return d; })();

/** 단면 위의 점: 경로면 α ∈ [0, 1], 방향이면 t ∈ [−1, 1] */
export function point(mode: Mode, s: number, dir: Float64Array) {
  const th = new Float64Array(D);
  if (mode === "line") { th[0] = -1 + 2 * s; return th; }
  if (mode === "curve") { const [u, v] = onValley(s); th[0] = u; th[1] = v; return th; }
  for (let i = 0; i < D; i++) th[i] = A[i] + s * dir[i];
  return th;
}
export function profile(mode: Mode, dir: Float64Array, K = 200) {
  const isPath = mode === "line" || mode === "curve";
  const ss = Array.from({ length: K + 1 }, (_, i) => (isPath ? i / K : -1 + (2 * i) / K));
  const ls = ss.map((s) => loss(point(mode, s, dir)));
  const ends = isPath ? Math.max(ls[0], ls[K]) : ls[K / 2];
  const peak = Math.max(...ls);
  return { ss, ls, peak, barrier: peak - ends };
}

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 520;
  const f = frame(root, "손실 단면과 보간 경로", "manipulate", narrow ? 0.95 : 0.5, "합성 손실의 (u, v) 평면 등고선과 두 해를 잇는 경로, 그리고 고른 경로나 방향을 따라 자른 1차원 손실 단면");
  let mode: Mode = "line", s = 0.5, seed = 7;
  let randDir = randomDir(seed), dir = randDir;
  let prof = profile(mode, dir);
  const LMAX = 1.25;
  let heat: HTMLCanvasElement | null = null;

  function boxes() {
    const W = f.width, H = f.height;
    if (!narrow) {
      const mw = Math.min(W * 0.44, (H - 24) * (3.2 / 2.6));
      return { map: { x: 10, y: 10, w: mw, h: H - 24 }, pr: { x: mw + 54, y: 12, w: W - mw - 66, h: H - 40 } };
    }
    const mh = Math.round((H - 50) * 0.5);
    return { map: { x: 10, y: 10, w: W - 20, h: mh }, pr: { x: 40, y: mh + 30, w: W - 52, h: H - mh - 58 } };
  }
  const U0 = -1.6, U1 = 1.6, V0 = -1.3, V1 = 1.3;
  function mapXf(map: { x: number; y: number; w: number; h: number }) {
    const sc = Math.min(map.w / (U1 - U0), map.h / (V1 - V0));
    const ox = map.x + (map.w - sc * (U1 - U0)) / 2, oy = map.y + (map.h - sc * (V1 - V0)) / 2;
    return { sc, ox, oy, MX: (u: number) => ox + (u - U0) * sc, MY: (v: number) => oy + (V1 - v) * sc };
  }
  function buildHeat(wpx: number, hpx: number) {
    const n = Math.max(40, Math.min(220, Math.round(wpx))), m = Math.max(30, Math.round((n * hpx) / wpx));
    const c = document.createElement("canvas"); c.width = n; c.height = m;
    const g = c.getContext("2d")!, img = g.createImageData(n, m);
    const [r1, g1, b1] = rgbOf(C.ink2), [r2, g2, b2] = rgbOf(C.mid);
    const LV = [0.05, 0.25, 1, 3];
    const band = (v: number) => LV.findIndex((l) => v < l);
    const val: number[][] = [];
    for (let j = 0; j < m; j++) { val.push([]); for (let i = 0; i < n; i++) val[j].push(Lp(U0 + ((i + 0.5) / n) * (U1 - U0), V1 - ((j + 0.5) / m) * (V1 - V0))); }
    for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) {
      const v = val[j][i], b = band(v), k = 4 * (j * n + i);
      const edge = (i + 1 < n && band(val[j][i + 1]) !== b) || (j + 1 < m && band(val[j + 1][i]) !== b);
      if (edge) { img.data[k] = r2; img.data[k + 1] = g2; img.data[k + 2] = b2; img.data[k + 3] = 120; continue; }
      const t = Math.min(1, Math.log1p(v * 10) / Math.log1p(30));
      img.data[k] = r1; img.data[k + 1] = g1; img.data[k + 2] = b1; img.data[k + 3] = Math.round(190 * (1 - t) ** 2);
    }
    g.putImageData(img, 0, 0);
    return c;
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { map, pr } = boxes();
    const { sc, MX, MY } = mapXf(map);
    const mw = sc * (U1 - U0), mh = sc * (V1 - V0);
    if (!heat || heat.dataset.key !== `${Math.round(mw)}x${C.ink2}`) { heat = buildHeat(mw, mh); heat.dataset.key = `${Math.round(mw)}x${C.ink2}`; }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, MX(U0), MY(V1), mw, mh);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(MX(U0), MY(V1), mw, mh);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    ctx.fillText("u", MX(U1) - 12, MY(0) - 5); ctx.fillText("v", MX(0) + 5, MY(V1) + 13);
    ctx.fillText("(u, v) 평면", MX(U0) + 6, MY(V0) - 6);
    const isPath = mode === "line" || mode === "curve";
    // 직선과 곡선 경로
    const paths: [Mode, () => void][] = [
      ["line", () => { ctx.moveTo(MX(-1), MY(0)); ctx.lineTo(MX(1), MY(0)); }],
      ["curve", () => VALLEY.pts.forEach(([u, v], i) => (i ? ctx.lineTo(MX(u), MY(v)) : ctx.moveTo(MX(u), MY(v))))],
    ];
    for (const [k, trace] of paths) {
      ctx.strokeStyle = k === mode ? C.edge : C.mid; ctx.lineWidth = k === mode ? 2.6 : 1.2;
      if (k !== mode) ctx.setLineDash([4, 4]);
      ctx.beginPath(); trace(); ctx.stroke(); ctx.setLineDash([]);
    }
    // 방향 단면이면 방향의 (u, v) 성분을 선분으로
    if (!isPath) {
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(MX(-1 - dir[0]), MY(-dir[1])); ctx.lineTo(MX(-1 + dir[0]), MY(dir[1])); ctx.stroke();
    }
    // 두 해
    for (const [u, name] of [[-1, "θ_a"], [1, "θ_b"]] as [number, string][]) {
      ctx.fillStyle = C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(MX(u), MY(0), 5, 0, 7); ctx.fill(); ctx.stroke();
      ctx.font = "600 12px JetBrains Mono, monospace";
      ctx.fillText(name, MX(u) + (u < 0 ? 6 : -36), MY(0) + 18);
    }
    // 현재 점
    const th = point(mode, s, dir);
    ctx.fillStyle = C.edge; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(MX(th[0]), MY(th[1]), 5.5, 0, 7); ctx.fill(); ctx.stroke();

    // 1D 단면
    const s0 = isPath ? 0 : -1, s1 = 1;
    const PX = (v: number) => pr.x + ((v - s0) / (s1 - s0)) * pr.w;
    const PY = (l: number) => pr.y + ((LMAX - Math.min(l, LMAX + 0.3)) / LMAX) * pr.h;
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(pr.x, pr.y, pr.w, pr.h);
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(pr.x, PY(v)); ctx.lineTo(pr.x + pr.w, PY(v)); ctx.stroke(); ctx.fillText(String(v), pr.x - 5, PY(v) + 4); }
    ctx.textAlign = "center";
    const ticks = isPath ? [0, 0.5, 1] : [-1, 0, 1];
    ticks.forEach((v) => ctx.fillText(String(v).replace("-", "−"), PX(v), pr.y + pr.h + 14));
    ctx.textAlign = "start";
    ctx.fillText(mode === "line" ? "L((1−α)θ_a + αθ_b)" : mode === "curve" ? "L(γ(α)), γ = 골짜기 바닥 곡선" : "L(θ_a + t·d), |d| = 1", pr.x + 6, pr.y + 14);
    if (mode === "curve") ctx.fillText("α = 곡선 위 호 길이 비율", pr.x + 6, pr.y + 28);
    ctx.save(); ctx.beginPath(); ctx.rect(pr.x, pr.y, pr.w, pr.h); ctx.clip();
    if (isPath) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
      ctx.beginPath(); ctx.moveTo(pr.x, PY(prof.peak)); ctx.lineTo(pr.x + pr.w, PY(prof.peak)); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4; ctx.lineJoin = "round";
    ctx.beginPath(); prof.ss.forEach((v, i) => (i ? ctx.lineTo(PX(v), PY(prof.ls[i])) : ctx.moveTo(PX(v), PY(prof.ls[i])))); ctx.stroke();
    const lnow = loss(th);
    ctx.fillStyle = C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(PX(s), PY(lnow), 5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function update() {
    prof = profile(mode, dir);
    const isPath = mode === "line" || mode === "curve";
    const lnow = loss(point(mode, s, dir));
    if (isPath) {
      ro.set("a", `L(θ_a) ${loss(point(mode, 0, dir)).toFixed(3)} · L(θ_b) ${loss(point(mode, 1, dir)).toFixed(3)}`);
      ro.set("b", `경로 최대 ${prof.peak.toFixed(3)} · 장벽 ${prof.barrier.toFixed(3)}`, prof.barrier > 0.2 ? "hot" : "ok");
      ro.set("c", `α = ${s.toFixed(2)} · L = ${lnow.toFixed(3)}`);
    } else {
      const k = curvature(dir), pl = Math.hypot(dir[0], dir[1]);
      ro.set("a", `곡률 dᵀHd = ${k.toFixed(2)} · λ_max = ${HDIAG[2]}`, k > 10 ? "hot" : "");
      ro.set("b", `d의 (u, v) 평면 성분 ${pl.toFixed(2)}`, "");
      ro.set("c", `t = ${s.toFixed(2)} · L = ${lnow.toFixed(3)}`);
    }
    pageState.set("lossSlice", { mode, s }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const SHORT: Record<Mode, string> = { line: "직선", curve: "곡선", rand: "무작위", top: "최대 곡률" };
  const pre = presets<Mode>(narrow ? MODES.map((m) => ({ key: m.key, label: SHORT[m.key] })) : MODES, (k) => {
    const wasPath = mode === "line" || mode === "curve", isPath = k === "line" || k === "curve";
    mode = k;
    dir = k === "top" ? TOP : randDir;
    reroll.hidden = k !== "rand";
    if (wasPath !== isPath) {
      ps.input.min = isPath ? "0" : "-1"; ps.input.step = isPath ? "0.01" : "0.02";
      s = 0.5; ps.set(s);
      ps.row.querySelector("label")!.textContent = isPath ? "α" : "t";
      ps.input.setAttribute("aria-label", isPath ? "경로 위 위치 α" : "방향 단면의 거리 t");
    }
    update();
  }, mode);
  const ps = param({ label: "α", min: 0, max: 1, step: 0.01, value: s, aria: "경로 위 위치 α", onInput: (v) => { s = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(ps.row);
  const tools = el("div", "wframe-row");
  const reroll = el("button", "preset");
  reroll.type = "button";
  reroll.textContent = "다른 무작위 방향";
  reroll.hidden = true;
  reroll.addEventListener("click", () => { seed += 1; dir = randDir = randomDir(seed); update(); });
  tools.append(reroll);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
