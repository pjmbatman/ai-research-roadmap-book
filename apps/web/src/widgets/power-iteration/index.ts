// 반복 적용과 고유벡터 (data/widgets.toml: power-iteration, 7.5장 "고윳값과 고유벡터").
// 거듭제곱법 v_{k+1} = A v_k / ‖A v_k‖를 2×2 행렬에 적용해, v_k가 지배 고유벡터 v₁ 쪽으로 도는 모습과
// v₁과의 각 오차 sin θ_k가 대략 |λ₂/λ₁|^k로 줄어드는 것을 보인다.
// 조작: 행렬 프리셋(본문의 인구 이동 행렬 등), 고윳값 비 λ₂/λ₁ 슬라이더(고유벡터와 λ₁은 그대로 두고 λ₂만 바꾼다),
// 반복 횟수 k, 그림에서 시작 벡터 v₀ 끌기.
import { frame, param, presets, readouts, el, C, reducedMotion } from "../frame";

export type M2 = [number, number, number, number]; // a11 a12 a21 a22
export type V2 = [number, number];

/** 실고윳값 2×2 행렬의 고유분해. |λ| 큰 순서. 복소수면 null. */
export function eig2(A: M2): { lam: [number, number]; V: [V2, V2] } | null {
  const [a, b, c, d] = A;
  const tr = a + d, det = a * d - b * c;
  const disc = tr * tr / 4 - det;
  if (disc < -1e-12) return null;
  const s = Math.sqrt(Math.max(disc, 0));
  let l1 = tr / 2 + s, l2 = tr / 2 - s;
  if (Math.abs(l2) > Math.abs(l1)) [l1, l2] = [l2, l1];
  const vec = (l: number): V2 => {
    // (A − λI)v = 0의 해: 행 (a−λ, b)에 수직인 (b, λ−a) 또는 행 (c, d−λ)에 수직인 (λ−d, c)
    const u: V2 = [b, l - a], w: V2 = [l - d, c];
    const v = Math.hypot(...u) >= Math.hypot(...w) ? u : w;
    const n = Math.hypot(...v);
    if (n < 1e-12) return Math.abs(l - a) < 1e-12 && Math.abs(b) < 1e-12 && Math.abs(c) < 1e-12 ? [1, 0] : [0, 1];
    return [v[0] / n, v[1] / n];
  };
  let v1 = vec(l1), v2 = vec(l2);
  if (Math.abs(l1 - l2) < 1e-12) { v1 = [1, 0]; v2 = [0, 1]; }
  return { lam: [l1, l2], V: [v1, v2] };
}

/** V diag(λ₁, λ₂) V⁻¹ (V의 열 = 고유벡터) */
export function compose(v1: V2, v2: V2, l1: number, l2: number): M2 {
  const det = v1[0] * v2[1] - v2[0] * v1[1];
  const inv = [v2[1] / det, -v2[0] / det, -v1[1] / det, v1[0] / det];
  const VL = [v1[0] * l1, v2[0] * l2, v1[1] * l1, v2[1] * l2];
  return [
    VL[0] * inv[0] + VL[1] * inv[2], VL[0] * inv[1] + VL[1] * inv[3],
    VL[2] * inv[0] + VL[3] * inv[2], VL[2] * inv[1] + VL[3] * inv[3],
  ];
}

const mul = (A: M2, v: V2): V2 => [A[0] * v[0] + A[1] * v[1], A[2] * v[0] + A[3] * v[1]];

/** 거듭제곱법 v₀…v_K (각각 단위 벡터)와 정규화 전 A v_{k−1} */
export function powerIterate(A: M2, v0: V2, K: number) {
  const vs: V2[] = [v0];
  const raw: V2[] = [v0];
  for (let k = 0; k < K; k++) {
    const w = mul(A, vs[k]);
    const n = Math.hypot(...w) || 1;
    raw.push(w);
    vs.push([w[0] / n, w[1] / n]);
  }
  return { vs, raw };
}
/** v와 직선 span(u) 사이 각의 sin (u는 단위 벡터) */
export const sinTo = (v: V2, u: V2) => Math.abs(v[0] * u[1] - v[1] * u[0]) / (Math.hypot(...v) || 1);
/** Rayleigh quotient vᵀAv / vᵀv */
export const rayleigh = (A: M2, v: V2) => { const w = mul(A, v); return (v[0] * w[0] + v[1] * w[1]) / (v[0] * v[0] + v[1] * v[1]); };

const PRESETS: Record<string, { label: string; A: M2 }> = {
  pop: { label: "인구 이동(본문)", A: [0.8, 0.3, 0.2, 0.7] },
  sym: { label: "대칭 [[2,1],[1,2]]", A: [2, 1, 1, 2] },
  close: { label: "λ 비슷함", A: [0.95, 0.05, 0.05, 0.95] },
  neg: { label: "λ₂ 음수", A: [0.2, 0.8, 0.8, 0.2] },
  nonsym: { label: "비대칭 [[2,1],[0,3]]", A: [2, 1, 0, 3] },
};
const KMAX = 30;
const FLOOR = 1e-10;

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = root.clientWidth < 560;
  const f = frame(root, "반복 적용과 고유벡터", "manipulate", narrow ? 0.92 : 0.52,
    "단위원 위의 반복 벡터 v_k와 두 고유벡터 방향, 그리고 v₁과의 각 오차가 k에 따라 줄어드는 로그 그래프");
  const startKey = attrs.preset in PRESETS ? attrs.preset : "pop";
  let A: M2 = [...PRESETS[startKey].A];
  let k = 8;
  let th0 = (160 * Math.PI) / 180; // 시작 벡터 각
  let drag = false;

  const ed = () => eig2(A)!;

  function layout() {
    const W = f.width, H = f.height;
    if (W < 560) {
      const side = Math.min(W - 16, H * 0.52);
      return { cx: W / 2, cy: 8 + side / 2, R: side * 0.36, plot: { x: 46, y: side + 26, w: W - 62, h: H - side - 70 } };
    }
    const side = Math.min(H - 16, W * 0.5);
    return { cx: 8 + side / 2, cy: H / 2, R: side * 0.36, plot: { x: side + 56, y: 22, w: W - side - 72, h: H - 66 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const L = layout();
    const { cx, cy, R } = L;
    const P = (x: number, y: number): V2 => [cx + x * R, cy - y * R];
    const { lam, V } = ed();
    const { vs, raw } = powerIterate(A, [Math.cos(th0), Math.sin(th0)], KMAX);
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    line(P(-1.35, 0), P(1.35, 0)); line(P(0, -1.35), P(0, 1.35));
    // 단위원
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.strokeStyle = C.grid2; ctx.stroke();
    // 고유벡터 방향(직선)
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = C.ink2;
    line(P(-1.3 * V[1][0], -1.3 * V[1][1]), P(1.3 * V[1][0], 1.3 * V[1][1]));
    ctx.strokeStyle = C.edge;
    line(P(-1.3 * V[0][0], -1.3 * V[0][1]), P(1.3 * V[0][0], 1.3 * V[0][1]));
    ctx.setLineDash([]);
    ctx.font = "600 12px JetBrains Mono, monospace";
    const lab = (v: V2, s: string, color: string) => {
      const sgn = v[1] < -1e-9 || (Math.abs(v[1]) < 1e-9 && v[0] < 0) ? -1 : 1;
      const q = P(1.32 * v[0] * sgn, 1.32 * v[1] * sgn);
      ctx.fillStyle = color; ctx.fillText(s, q[0] + 4, q[1] - 4);
    };
    lab(V[0], "v₁", C.edge); lab(V[1], "v₂", C.mid);
    // 이전 반복: 점과 번호
    for (let j = 0; j < k; j++) {
      const q = P(...vs[j]);
      ctx.fillStyle = C.edgeSoft;
      dot(q, 4);
      if (j < 4) { ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace"; ctx.fillText(String(j), q[0] + 6, q[1] + 12); }
    }
    // 정규화 전 A v_{k−1} (짧게 줄인 길이로)
    if (k > 0) {
      const w = raw[k];
      const n = Math.hypot(...w);
      const s = Math.min(1.3, n) / (n || 1);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
      line(P(0, 0), P(w[0] * s, w[1] * s));
      ctx.setLineDash([]);
    }
    // 현재 v_k
    arrow(P(0, 0), P(...vs[k]), C.edge);
    ctx.fillStyle = C.edge; ctx.font = "700 12px JetBrains Mono, monospace";
    const vk = P(...vs[k]);
    ctx.fillText(`v${sub(k)}`, vk[0] + 8, vk[1] - 8);
    // 시작 벡터 손잡이
    const h0 = P(Math.cos(th0), Math.sin(th0));
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(h0[0], h0[1], drag ? 9 : 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 1;

    // 오차 그래프: log10 sin θ_k
    const pl = L.plot;
    const yOf = (e: number) => pl.y + (-Math.log10(Math.max(e, FLOOR)) / 10) * pl.h;
    const xOf = (j: number) => pl.x + (j / KMAX) * pl.w;
    ctx.strokeStyle = C.faint;
    ctx.fillStyle = C.mid;
    ctx.font = "11px JetBrains Mono, monospace";
    for (let e = 0; e <= 10; e += 2) {
      const y = yOf(10 ** -e);
      line([pl.x, y], [pl.x + pl.w, y]);
      ctx.fillText(e === 0 ? "1" : `1e−${e}`, pl.x - 40, y + 4);
    }
    for (let j = 0; j <= KMAX; j += 10) ctx.fillText(String(j), xOf(j) - 4, pl.y + pl.h + 16);
    ctx.fillText("k", pl.x + pl.w - 4, pl.y + pl.h + 30);
    ctx.fillStyle = C.ink;
    ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText("sin θ_k  (v_k와 v₁ 사이)", pl.x, pl.y - 8);
    // 기준선 sin θ₀ · |λ₂/λ₁|^k (대칭이면 tan θ_k = tan θ₀ · |r|^k)
    const r = Math.abs(lam[1] / lam[0]);
    const e0 = Math.max(sinTo(vs[0], V[0]), 1e-12);
    ctx.save();
    ctx.beginPath(); ctx.rect(pl.x, pl.y, pl.w, pl.h); ctx.clip();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = C.mid;
    ctx.beginPath();
    for (let j = 0; j <= KMAX; j++) {
      const e = e0 * r ** j;
      const [x, y] = [xOf(j), yOf(e)];
      if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      if (e < FLOOR) break;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    for (let j = 0; j <= KMAX; j++) {
      const e = sinTo(vs[j], V[0]);
      if (e < FLOOR) break;
      ctx.fillStyle = j <= k ? C.edge : C.edgeSoft;
      dot([xOf(j), yOf(e)], j === k ? 5 : 3);
    }
    ctx.fillStyle = C.mid;
    ctx.font = "11px JetBrains Mono, monospace";
    const tag = `- - sin θ₀·|λ₂/λ₁|^k`;
    ctx.fillText(tag, pl.x + 6, pl.y + pl.h - 6);
  }
  const sub = (n: number) => String(n).split("").map((c) => "₀₁₂₃₄₅₆₇₈₉"[Number(c)]).join("");
  function line(a: V2, c: V2) { f.ctx.beginPath(); f.ctx.moveTo(...a); f.ctx.lineTo(...c); f.ctx.stroke(); }
  function dot(p: V2, r: number) { f.ctx.beginPath(); f.ctx.arc(p[0], p[1], r, 0, 7); f.ctx.fill(); }
  function arrow(a: V2, c: V2, color: string) {
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
  }

  // 조작
  const pre = presets(Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label })), (key) => {
    A = [...PRESETS[key].A]; syncRatio(); update();
  }, startKey);
  const fmtR = (v: number) => (v >= 0 ? " " : "") + v.toFixed(2);
  const pr = param({
    label: "λ₂/λ₁", min: -0.95, max: 0.95, step: 0.01, value: 0, fmt: fmtR, aria: "둘째 고윳값과 지배 고윳값의 비",
    onInput: (v) => {
      const { lam, V } = ed();
      A = compose(V[0], V[1], lam[0], v * lam[0]);
      pre.select(null);
      update();
    },
  });
  const pk = param({ label: "k", min: 0, max: KMAX, step: 1, value: k, fmt: (v) => String(v), aria: "거듭제곱법 반복 횟수", onInput: (v) => { k = v; update(); } });
  pr.row.style.gridTemplateColumns = "3.4em minmax(0, 1fr) 3.6em";
  const grid = el("div", "param-grid");
  grid.append(pr.row, pk.row);
  const tools = el("div", "wframe-row");
  const play = el("button", "preset");
  play.type = "button";
  play.textContent = "▶ k = 0부터 반복";
  play.addEventListener("click", animate);
  tools.append(play);
  const ro = readouts();
  pre.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(pre.row, grid, tools, ro.row);

  function syncRatio() { const { lam } = ed(); pr.set(Math.round((lam[1] / lam[0]) * 100) / 100); }
  function update() {
    const { lam, V } = ed();
    const { vs } = powerIterate(A, [Math.cos(th0), Math.sin(th0)], k);
    const r = Math.abs(lam[1] / lam[0]);
    const e = sinTo(vs[k], V[0]);
    const n = (v: number) => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(2);
    ro.set("A", `A = [[${n(A[0])}, ${n(A[1])}], [${n(A[2])}, ${n(A[3])}]]`);
    ro.set("lam", `λ₁ = ${n(lam[0])}, λ₂ = ${n(lam[1])}`);
    ro.set("ray", `k = ${k}: vᵀAv = ${rayleigh(A, vs[k]).toFixed(5)}`, "hot");
    ro.set("err", `sin θ_k = ${e < 1e-15 ? "0" : e.toExponential(1)}`, e < 1e-6 ? "ok" : "");
    ro.set("rate", r < 1e-9 ? "|λ₂/λ₁| = 0 · 한 번에 수렴" : `|λ₂/λ₁| = ${r.toFixed(2)} · 10배 줄이는 데 ≈ ${(-1 / Math.log10(r)).toFixed(1)}번`);
    draw();
  }
  function animate() {
    if (reducedMotion()) { k = KMAX; pk.set(k); update(); return; }
    let j = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const target = Math.min(KMAX, Math.floor((now - t0) / 120));
      if (target !== j) { j = target; k = j; pk.set(k); update(); }
      if (j < KMAX) requestAnimationFrame(step);
    };
    k = 0; pk.set(0); update();
    requestAnimationFrame(step);
  }

  // 끌기: 시작 벡터 v₀를 단위원 위에서 돌린다
  const pos = (e: PointerEvent) => { const r = f.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] as V2; };
  const near = (x: number, y: number) => {
    const { cx, cy, R } = layout();
    const h: V2 = [cx + Math.cos(th0) * R, cy - Math.sin(th0) * R];
    return Math.hypot(h[0] - x, h[1] - y) < 20;
  };
  f.canvas.addEventListener("pointerdown", (e) => {
    const [x, y] = pos(e);
    if (near(x, y)) { drag = true; f.canvas.setPointerCapture(e.pointerId); }
  });
  f.canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    if (!drag) { f.canvas.style.cursor = near(x, y) ? "grab" : "default"; return; }
    const { cx, cy } = layout();
    th0 = Math.atan2(cy - y, x - cx);
    update();
  });
  const end = () => { if (drag) { drag = false; draw(); } };
  f.canvas.addEventListener("pointerup", end);
  f.canvas.addEventListener("pointercancel", end);

  syncRatio();
  f.onResize(draw);
  update();
}
