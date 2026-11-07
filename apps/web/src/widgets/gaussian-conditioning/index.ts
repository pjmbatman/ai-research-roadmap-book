// 다변량 가우시안의 조건부분포 (data/widgets.toml: gaussian-conditioning).
// 2차원 가우시안 (x₁, x₂) ~ N(0, Σ), Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]에서 x₁을 고정하면
//   x₂ | x₁ ~ N(μ₂ + ρ(σ₂/σ₁)(x₁ − μ₁), σ₂²(1 − ρ²)).
// 왼쪽: 등고선(마할라노비스 거리 1, 2), 회귀선 E[x₂|x₁], 고정한 x₁의 세로선. 오른쪽: 주변분포 p(x₂)와 조건부분포 p(x₂|x₁).
// 표본을 켜면 x₁ 근처 띠 안의 표본으로 같은 조건부분포를 추정해 공식과 비교한다.
import { frame, param, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const S1 = 1, S2 = 1.5, M1 = 0, M2 = 0;
/** 조건부 평균과 분산 (블록 공식) */
export function conditional(rho: number, x1: number) {
  const s12 = rho * S1 * S2;
  return { mean: M2 + (s12 / (S1 * S1)) * (x1 - M1), variance: S2 * S2 - (s12 * s12) / (S1 * S1) };
}
const NS = 4000, BAND = 0.15;
const gauss = (x: number, m: number, v: number) => Math.exp((-0.5 * (x - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v);

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const f = frame(root, "다변량 가우시안의 조건부분포", "manipulate", 0.66, "2차원 가우시안의 등고선과 표본, 고정한 x₁에서의 조건부분포와 주변분포");
  let rho = Number(attrs.rho ?? 0.8);
  if (!Number.isFinite(rho)) rho = 0.8;
  let x1 = 1.2, showSamples = true;
  const r = rng(20261107);
  const Z = Array.from({ length: NS }, () => [r.normal(), r.normal()] as [number, number]);
  let pts: [number, number][] = [];
  let band = { n: 0, mean: NaN, variance: NaN };

  const XR = 3.5, YR = 4.5;
  const geo = () => {
    const W = f.width, H = f.height;
    const side = Math.max(96, Math.min(180, W * 0.28));
    const pl = 34, pr = W - side - 14, pt = 34, pb = H - 24;
    return { pl, pr, pt, pb, sl: pr + 14, sr: W - 10 };
  };
  const PX = (v: number) => { const g = geo(); return g.pl + ((v + XR) / (2 * XR)) * (g.pr - g.pl); };
  const PY = (v: number) => { const g = geo(); return g.pb - ((v + YR) / (2 * YR)) * (g.pb - g.pt); };

  function compute() {
    const c = Math.sqrt(1 - rho * rho);
    pts = Z.map(([a, b]) => [M1 + S1 * a, M2 + S2 * (rho * a + c * b)]);
    const sel = pts.filter(([a]) => Math.abs(a - x1) <= BAND).map(([, b]) => b);
    const n = sel.length;
    const mean = n ? sel.reduce((s, v) => s + v, 0) / n : NaN;
    const variance = n > 1 ? sel.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : NaN;
    band = { n, mean, variance };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    const g = geo();
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    // 격자
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (let k = -3; k <= 3; k++) {
      ctx.beginPath(); ctx.moveTo(PX(k), g.pt); ctx.lineTo(PX(k), g.pb); ctx.stroke();
      ctx.fillText(String(k), PX(k) - (k < 0 ? 7 : 3), g.pb + 14);
    }
    for (let k = -4; k <= 4; k += 2) {
      ctx.beginPath(); ctx.moveTo(g.pl, PY(k)); ctx.lineTo(g.pr, PY(k)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(g.sl, PY(k)); ctx.lineTo(g.sr, PY(k)); ctx.stroke();
      ctx.fillText(String(k), k < 0 ? 4 : 10, PY(k) + 4);
    }
    ctx.fillText("x₁", g.pr - 16, g.pb - 6);
    ctx.fillText("x₂", g.pl + 6, g.pt - 8);

    // 표본
    const cond = conditional(rho, x1);
    ctx.save();
    ctx.beginPath(); ctx.rect(g.pl, g.pt, g.pr - g.pl, g.pb - g.pt); ctx.clip();
    if (showSamples) {
      ctx.fillStyle = alpha(C.mid, 0.35);
      for (const [a, b] of pts) if (Math.abs(a - x1) > BAND) ctx.fillRect(PX(a) - 1, PY(b) - 1, 2, 2);
      ctx.fillStyle = C.edge;
      for (const [a, b] of pts) if (Math.abs(a - x1) <= BAND) ctx.fillRect(PX(a) - 1.5, PY(b) - 1.5, 3, 3);
    }
    // 등고선: 마할라노비스 거리 1, 2
    const c = Math.sqrt(1 - rho * rho);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
    for (const rad of [1, 2]) {
      ctx.beginPath();
      for (let i = 0; i <= 96; i++) {
        const t = (2 * Math.PI * i) / 96, a = rad * Math.cos(t), b = rad * Math.sin(t);
        const X = PX(M1 + S1 * a), Y = PY(M2 + S2 * (rho * a + c * b));
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      ctx.stroke();
    }
    // 회귀선 E[x₂ | x₁]
    ctx.setLineDash([6, 5]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(PX(-XR), PY(conditional(rho, -XR).mean)); ctx.lineTo(PX(XR), PY(conditional(rho, XR).mean)); ctx.stroke();
    ctx.setLineDash([]);
    // 고정한 x₁과 조건부 평균 ± 2σ
    ctx.strokeStyle = alpha(C.edge, 0.55); ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(PX(x1), g.pt); ctx.lineTo(PX(x1), g.pb); ctx.stroke();
    if (!showSamples) {
      // 표본이 없을 때는 조건부 평균 ± 2σ 구간을 세로선 위에 굵게 표시한다
      const sd = Math.sqrt(cond.variance);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 5; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(PX(x1), PY(cond.mean - 2 * sd)); ctx.lineTo(PX(x1), PY(cond.mean + 2 * sd)); ctx.stroke();
      ctx.lineCap = "butt";
    }
    ctx.fillStyle = C.bg; ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(PX(x1), PY(cond.mean), 6, 0, 7); ctx.fill(); ctx.stroke();
    ctx.restore();

    // 오른쪽: p(x₂)와 p(x₂ | x₁)
    const dmax = 0.9;
    const DX = (d: number) => g.sl + (Math.min(d, dmax) / dmax) * (g.sr - g.sl);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(g.sl, g.pt); ctx.lineTo(g.sl, g.pb); ctx.stroke();
    if (showSamples && band.n > 1) {
      // 띠 안 표본의 히스토그램(밀도로 정규화)
      const bw = 0.3, counts = new Map<number, number>();
      for (const [a, b] of pts) if (Math.abs(a - x1) <= BAND) { const k = Math.floor(b / bw); counts.set(k, (counts.get(k) ?? 0) + 1); }
      ctx.fillStyle = C.fill;
      counts.forEach((cnt, k) => { const y0 = PY((k + 1) * bw), y1 = PY(k * bw); ctx.fillRect(g.sl, y0, DX(cnt / (band.n * bw)) - g.sl, y1 - y0 - 1); });
    }
    const curve = (m: number, v: number) => {
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) { const y = -YR + (2 * YR * i) / 160; i ? ctx.lineTo(DX(gauss(y, m, v)), PY(y)) : ctx.moveTo(DX(gauss(y, m, v)), PY(y)); }
    };
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5;
    curve(M2, S2 * S2); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    curve(cond.mean, cond.variance); ctx.stroke();
    ctx.strokeStyle = alpha(C.edge, 0.5); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(g.sl, PY(cond.mean)); ctx.lineTo(DX(gauss(cond.mean, cond.mean, cond.variance)), PY(cond.mean)); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText("p(x₂|x₁)", g.sl, 14);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("p(x₂)", g.sl, 28);
  }

  function update() {
    compute();
    const cond = conditional(rho, x1);
    ro.set("mean", `E[x₂|x₁] = ${cond.mean.toFixed(3)}`, "hot");
    ro.set("var", `Var[x₂|x₁] = ${cond.variance.toFixed(3)} (주변 ${(S2 * S2).toFixed(2)})`, "hot");
    ro.set("band", showSamples
      ? `x₁ ≈ ${x1.toFixed(2)} 표본 ${band.n}개 · 평균 ${Number.isNaN(band.mean) ? "–" : band.mean.toFixed(2)} · 분산 ${Number.isNaN(band.variance) ? "–" : band.variance.toFixed(2)}`
      : `σ₁ = ${S1}, σ₂ = ${S2}, 공분산 ${(rho * S1 * S2).toFixed(2)}`);
    pageState.set("rho", rho, root);
    pageState.set("x1", x1, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }
  let raf = 0;

  const pr = param({ label: "ρ", min: -0.95, max: 0.95, step: 0.05, value: rho, aria: "상관계수 ρ", onInput: (v) => { rho = v; update(); } });
  const px = param({ label: "x₁", min: -3, max: 3, step: 0.05, value: x1, aria: "고정할 x₁ 값", onInput: (v) => { x1 = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pr.row, px.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" checked /> 표본 ${NS}개`;
  sw.querySelector("input")!.addEventListener("change", (e) => { showSamples = (e.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);

  let dragging = false;
  const setX = (e: PointerEvent) => {
    const rect = f.canvas.getBoundingClientRect(), g = geo();
    const v = -XR + ((e.clientX - rect.left - g.pl) / (g.pr - g.pl)) * 2 * XR;
    x1 = Math.round(Math.max(-3, Math.min(3, v)) * 20) / 20;
    px.set(x1); update();
  };
  f.canvas.addEventListener("pointerdown", (e) => {
    const rect = f.canvas.getBoundingClientRect();
    if (e.clientX - rect.left > geo().pr) return;
    dragging = true; f.canvas.setPointerCapture(e.pointerId); setX(e);
  });
  f.canvas.addEventListener("pointermove", (e) => {
    if (dragging) { setX(e); return; }
    const rect = f.canvas.getBoundingClientRect();
    f.canvas.style.cursor = e.clientX - rect.left <= geo().pr ? "ew-resize" : "default";
  });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.onResize(draw);
  update();
  draw();
}
