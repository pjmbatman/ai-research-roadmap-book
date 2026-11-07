// 부트스트랩의 재표본화 단위 (data/widgets.toml: bootstrap-resampler).
// 12.4장 실습과 같은 구조의 가상 데이터: 셀 G = 20개 × 셀마다 m = 50 사이클, 샘플마다 두 모델의 오차 차이
//   d_ij = δ + σ (√ρ · a_i + √(1 − ρ) · e_ij),  δ = −0.10 %p, σ = 0.44 %p, a_i, e_ij ~ N(0, 1)
// 여기서 ρ는 그룹 상관(같은 셀 안 두 샘플의 차이가 닮은 정도, ICC)이다. 새 셀들에서 다시 실험했을 때 평균 차이의 실제 표준오차는
//   SE = σ √(ρ/G + (1 − ρ)/(G m)).
// 샘플 단위 부트스트랩은 1000개 샘플을, 셀 단위 부트스트랩은 셀 20개를 복원추출한다(B = 1000번, 고정 시드).
import { frame, param, presets, readouts, el, C, rng, alpha } from "../frame";
import { pageState } from "../../client/store";

export const G = 20, M = 50, NS = G * M, B = 1000, DELTA = -0.1, SIGMA = 0.44;
const RAW = (() => {
  const r = rng(2024);
  const a = Float64Array.from({ length: G }, () => r.normal());
  const e = Float64Array.from({ length: NS }, () => r.normal());
  const sIdx = new Uint16Array(B * NS), cIdx = new Uint8Array(B * G);
  for (let i = 0; i < sIdx.length; i++) sIdx[i] = Math.floor(r.u() * NS);
  for (let i = 0; i < cIdx.length; i++) cIdx[i] = Math.floor(r.u() * G);
  return { a, e, sIdx, cIdx };
})();

export function data(rho: number) {
  const d = new Float64Array(NS), cm = new Float64Array(G);
  const sa = Math.sqrt(rho), se = Math.sqrt(1 - rho);
  for (let i = 0; i < G; i++) {
    let s = 0;
    for (let j = 0; j < M; j++) { const v = DELTA + SIGMA * (sa * RAW.a[i] + se * RAW.e[i * M + j]); d[i * M + j] = v; s += v; }
    cm[i] = s / M;
  }
  return { d, cm, mean: cm.reduce((s, v) => s + v, 0) / G };
}
const sd = (a: Float64Array) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
const pct = (a: Float64Array, q: number) => { const s = Array.from(a).sort((u, v) => u - v); const k = (s.length - 1) * q, i = Math.floor(k); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (k - i); };

/** 두 단위의 부트스트랩 평균 분포와 실제 표준오차 */
export function bootstrap(rho: number) {
  const D = data(rho);
  const bs = new Float64Array(B), bc = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    const o = b * NS;
    for (let i = 0; i < NS; i++) s += D.d[RAW.sIdx[o + i]];
    bs[b] = s / NS;
    let c = 0;
    for (let g = 0; g < G; g++) c += D.cm[RAW.cIdx[b * G + g]];
    bc[b] = c / G;
  }
  const trueSE = SIGMA * Math.sqrt(rho / G + (1 - rho) / NS);
  return { ...D, bs, bc, seS: sd(bs), seC: sd(bc), trueSE };
}
const Phi = (z: number) => { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const p = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))) * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); return z >= 0 ? 1 - p : p; };

type Unit = "sample" | "cell";

export function mount(root: HTMLElement) {
  const f = frame(root, "부트스트랩 단위", "manipulate", 0.8, "셀 20개의 샘플별 오차 차이와, 샘플 단위·셀 단위 부트스트랩 분포 및 실제 표본분포의 폭");
  let rho = 0.2, unit: Unit = "sample";
  let res = bootstrap(rho);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 10, top = 20, base1 = Math.round(H * 0.46), top2 = base1 + 40, base2 = H - 22;
    // 위: 셀별 샘플(점)과 셀 평균(가로 막대)
    const lo = -1.6, hi = 1.4;
    const Y = (v: number) => base1 - ((v - lo) / (hi - lo)) * (base1 - top);
    const cw = (W - PL - PR) / G;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.fillStyle = C.mid;
    for (const v of [-1, 0, 1]) { ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke(); ctx.fillText((v > 0 ? "+" : "") + v, 4, Y(v) + 4); }
    ctx.fillStyle = alpha(C.ink2, 0.45);
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < M; j++) {
        const v = res.d[i * M + j];
        if (v < lo || v > hi) continue;
        const jx = ((j * 0.618034) % 1 - 0.5) * cw * 0.55;
        ctx.fillRect(PL + (i + 0.5) * cw + jx - 1, Y(v) - 1, 2, 2);
      }
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    for (let i = 0; i < G; i++) { const y = Y(res.cm[i]); ctx.beginPath(); ctx.moveTo(PL + (i + 0.12) * cw, y); ctx.lineTo(PL + (i + 0.88) * cw, y); ctx.stroke(); }
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(PL, Y(DELTA)); ctx.lineTo(W - PR, Y(DELTA)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.mid;
    ctx.fillText(`e_A − e_B [%p] · 셀 ${G}개 × ${M} (가상)`, PL, top - 6);
    // 아래: 부트스트랩 분포
    const span = 0.5;
    const c0 = res.mean;
    const X = (v: number) => PL + ((v - (c0 - span / 2)) / span) * (W - PL - PR);
    const NB = 60, bw = span / NB;
    const hist = (arr: Float64Array) => { const h = new Array(NB).fill(0); for (const v of arr) { const k = Math.floor((v - (c0 - span / 2)) / bw); if (k >= 0 && k < NB) h[k]++; } return h.map((c) => c / (B * bw)); };
    const hs = hist(res.bs), hc = hist(res.bc);
    const sel = unit === "sample" ? hs : hc, oth = unit === "sample" ? hc : hs;
    const pdf = (v: number) => Math.exp(-0.5 * ((v - c0) / res.trueSE) ** 2) / (res.trueSE * Math.sqrt(2 * Math.PI));
    const ymax = Math.max(...sel, ...oth, pdf(c0)) * 1.08;
    const Y2 = (v: number) => base2 - (v / ymax) * (base2 - top2);
    const pxw = (W - PL - PR) / NB;
    ctx.fillStyle = alpha(C.edge, 0.55);
    sel.forEach((v, k) => { if (v) ctx.fillRect(PL + k * pxw + 0.5, Y2(v), pxw - 1, base2 - Y2(v)); });
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath();
    oth.forEach((v, k) => { const x = PL + k * pxw; if (k === 0) ctx.moveTo(x, Y2(v)); else ctx.lineTo(x, Y2(v)); ctx.lineTo(x + pxw, Y2(v)); });
    ctx.stroke();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 160; i++) { const v = c0 - span / 2 + (span * i) / 160; i ? ctx.lineTo(X(v), Y2(pdf(v))) : ctx.moveTo(X(v), Y2(pdf(v))); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL, base2); ctx.lineTo(W - PR, base2); ctx.stroke();
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5;
    if (X(DELTA) > PL && X(DELTA) < W - PR) { ctx.beginPath(); ctx.moveTo(X(DELTA), top2); ctx.lineTo(X(DELTA), base2); ctx.stroke(); }
    ctx.fillStyle = C.mid;
    for (const v of [-0.3, -0.2, -0.1, 0, 0.1]) { const x = X(v); if (x < PL || x > W - PR - 10) continue; const s = v.toFixed(1); ctx.fillText(s, x - ctx.measureText(s).width / 2, base2 + 14); }
    // 범례
    const items: [string, string, "fill" | "step" | "dash"][] = [[unit === "sample" ? "샘플 단위" : "셀 단위", C.edge, "fill"], [unit === "sample" ? "셀 단위" : "샘플 단위", C.mid, "step"], ["실제 표본분포", C.ink, "dash"]];
    let lx = PL;
    for (const [name, col, kind] of items) {
      if (kind === "fill") { ctx.fillStyle = alpha(col, 0.55); ctx.fillRect(lx, top2 - 22, 14, 8); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.setLineDash(kind === "dash" ? [4, 3] : []); ctx.beginPath(); ctx.moveTo(lx, top2 - 18); ctx.lineTo(lx + 14, top2 - 18); ctx.stroke(); ctx.setLineDash([]); }
      ctx.fillStyle = C.mid; ctx.fillText(name, lx + 18, top2 - 14);
      lx += 30 + ctx.measureText(name).width;
    }
  }

  let raf = 0;
  function update() {
    res = bootstrap(rho);
    const arr = unit === "sample" ? res.bs : res.bc, se = unit === "sample" ? res.seS : res.seC;
    const lo = pct(arr, 0.025), hi = pct(arr, 0.975);
    const ratio = se / res.trueSE;
    ro.set("se", `SE 샘플 ${res.seS.toFixed(4)} · 셀 ${res.seC.toFixed(4)}`);
    ro.set("true", `실제 SE ${res.trueSE.toFixed(4)}`);
    ro.set("ratio", `${unit === "sample" ? "샘플" : "셀"} 단위 SE ÷ 실제 = ${ratio.toFixed(2)}`, ratio > 0.85 ? "ok" : "hot");
    ro.set("ci", `95% 구간 [${lo.toFixed(3)}, ${hi.toFixed(3)}]`, lo <= DELTA && DELTA <= hi ? "ok" : "");
    ro.set("cov", `δ 포함 비율 ≈ ${(2 * Phi(1.96 * ratio) - 1).toFixed(2)} (정규 근사)`);
    pageState.set("bootstrap", { unit, rho, se }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets<Unit>([{ key: "sample", label: "샘플 단위" }, { key: "cell", label: "셀 단위" }], (k) => { unit = k; update(); }, unit);
  pre.row.classList.add("segmented");
  const pr = param({ label: "ρ", min: 0, max: 0.8, step: 0.01, value: rho, aria: "그룹 상관 ρ (같은 셀 안 샘플 차이의 상관)", onInput: (v) => { rho = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pr.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
