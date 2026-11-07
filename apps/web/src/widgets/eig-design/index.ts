// 측정 설계와 기대 정보 이득 (data/widgets.toml: eig-design). 25.3장의 감쇠 예: y = e^{−kt} + ε, ε ~ N(0, 0.05²),
// 감쇠율 k ~ LogNormal(log k_med, 0.5) [1/s]. 측정 시각 t 하나를 고른다.
// k가 1차원이고 y도 1차원이라 EIG = H[y] − H[y | k]를 수치 적분으로 직접 계산한다(장의 중첩 MC와 달리 편향이 없다).
// H[y | k] = ½ log(2πeσ²), p(y) = Σ_j w_j N(y; e^{−k_j t}, σ²)(k는 log k의 등간격 격자, w_j는 사전 가중치).
// 위: 사전분포 분위수에 해당하는 감쇠 곡선들과 측정 시각. 가운데: EIG(t). 오른쪽(좁은 화면에서는 아래): 참 k가 사전 중앙값일 때
// 잡음 없는 측정 하나로 얻는 사후분포(log k 축)와 사전분포.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export const SIGMA = 0.05, S_LOG = 0.5;
const NZ = 241, ZMAX = 5;
const ZS = Array.from({ length: NZ }, (_, j) => -ZMAX + (2 * ZMAX * j) / (NZ - 1));
const WZ = (() => { const w = ZS.map((z) => Math.exp(-0.5 * z * z)); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => v / s); })();
const kOf = (kmed: number, z: number) => kmed * Math.exp(S_LOG * z);

/** 측정 시각 t의 EIG [nat]와 기대 사후 표준편차 E_y[sd(k | y)] */
export function eigAt(kmed: number, t: number) {
  const mu = ZS.map((z) => Math.exp(-kOf(kmed, z) * t));
  const ks = ZS.map((z) => kOf(kmed, z));
  const lo = Math.min(...mu) - 7 * SIGMA, hi = Math.max(...mu) + 7 * SIGMA;
  const dy = SIGMA / 8, ny = Math.ceil((hi - lo) / dy);
  const c = 1 / (SIGMA * Math.sqrt(2 * Math.PI));
  let H = 0, esd = 0;
  for (let i = 0; i <= ny; i++) {
    const y = lo + i * dy;
    let p = 0, m1 = 0, m2 = 0;
    for (let j = 0; j < NZ; j++) {
      const u = (y - mu[j]) / SIGMA;
      if (u * u > 80) continue;
      const v = WZ[j] * c * Math.exp(-0.5 * u * u);
      p += v; m1 += v * ks[j]; m2 += v * ks[j] * ks[j];
    }
    if (p <= 1e-300) continue;
    H -= p * Math.log(p) * dy;
    const m = m1 / p;
    esd += p * Math.sqrt(Math.max(0, m2 / p - m * m)) * dy;
  }
  return { eig: H - 0.5 * Math.log(2 * Math.PI * Math.E * SIGMA * SIGMA), esd };
}
/** 사전분포 표준편차(로그정규) */
export const priorSd = (kmed: number) => kmed * Math.sqrt((Math.exp(S_LOG ** 2) - 1) * Math.exp(S_LOG ** 2));
/** 참 k = k_med에서 잡음 없이 잰 y로 얻는 사후분포(log k의 밀도) */
export function posterior(kmed: number, t: number) {
  const y = Math.exp(-kmed * t);
  const w = ZS.map((z, j) => WZ[j] * Math.exp(-0.5 * ((y - Math.exp(-kOf(kmed, z) * t)) / SIGMA) ** 2));
  const s = w.reduce((a, b) => a + b, 0);
  const dz = (2 * ZMAX) / (NZ - 1) * S_LOG;   // log k 간격
  return w.map((v) => v / s / dz);
}

export const TS = Array.from({ length: 100 }, (_, i) => 0.05 * (i + 1));   // 0.05 … 5 s
export function curve(kmed: number) { return TS.map((t) => eigAt(kmed, t)); }

type Key = "k05" | "k1" | "k2";
const PRIORS: Record<Key, { label: string; kmed: number }> = { k05: { label: "k ~ 0.5", kmed: 0.5 }, k1: { label: "k ~ 1 (본문)", kmed: 1 }, k2: { label: "k ~ 2", kmed: 2 } };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "측정 설계와 기대 정보 이득", "manipulate", narrow ? 1.1 : 0.6, "사전분포의 감쇠 곡선들과 측정 시각, 측정 시각에 따른 기대 정보 이득 곡선, 측정 하나로 얻는 사후분포");
  let key: Key = "k1", ti = 9; // t = 0.5 s
  const cache = new Map<Key, ReturnType<typeof curve>>();
  const eigs = () => { if (!cache.has(key)) cache.set(key, curve(PRIORS[key].kmed)); return cache.get(key)!; };

  function boxes() {
    const W = f.width, H = f.height;
    if (!narrow && W >= 400) {
      const pw = Math.min(220, W * 0.34);
      const lw = W - pw - 34 - 30;
      const h1 = Math.round((H - 58) * 0.5);
      return { fan: { x: 34, y: 10, w: lw, h: h1 }, eg: { x: 34, y: h1 + 32, w: lw, h: H - h1 - 58 }, po: { x: W - pw - 10, y: 10, w: pw, h: H - 36 } };
    }
    const h = Math.round((H - 100) / 3);
    return { fan: { x: 34, y: 10, w: W - 44, h }, eg: { x: 34, y: h + 34, w: W - 44, h }, po: { x: 34, y: 2 * h + 70, w: W - 44, h: H - 2 * h - 96 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { fan, eg, po } = boxes();
    const kmed = PRIORS[key].kmed, t = TS[ti], E = eigs();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
    const TX = (b: typeof fan, s: number) => b.x + (s / 5) * b.w;
    // 감쇠 곡선 묶음
    const FY = (v: number) => fan.y + (1.05 - v) / 1.1 * fan.h;
    ctx.strokeStyle = C.faint; ctx.strokeRect(fan.x, fan.y, fan.w, fan.h);
    ctx.save(); ctx.beginPath(); ctx.rect(fan.x, fan.y, fan.w, fan.h); ctx.clip();
    const qs = [-1.64, -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28, 1.64];
    ctx.strokeStyle = C.edgeSoft; ctx.lineWidth = 1.4;
    for (const z of qs) {
      const k = kOf(kmed, z);
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) { const s = (5 * i) / 100; i ? ctx.lineTo(TX(fan, s), FY(Math.exp(-k * s))) : ctx.moveTo(TX(fan, s), FY(Math.exp(-k * s))); }
      ctx.stroke();
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(TX(fan, t), fan.y); ctx.lineTo(TX(fan, t), fan.y + fan.h); ctx.stroke();
    // 측정 시각에서 곡선 값(점)과 잡음 ±2σ 막대
    ctx.fillStyle = C.ink2;
    for (const z of qs) { ctx.beginPath(); ctx.arc(TX(fan, t), FY(Math.exp(-kOf(kmed, z) * t)), 2.6, 0, 7); ctx.fill(); }
    const ym = Math.exp(-kmed * t), bx = TX(fan, t) + 10;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(bx, FY(ym + 2 * SIGMA)); ctx.lineTo(bx, FY(ym - 2 * SIGMA)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(bx - 3, FY(ym + 2 * SIGMA)); ctx.lineTo(bx + 3, FY(ym + 2 * SIGMA)); ctx.moveTo(bx - 3, FY(ym - 2 * SIGMA)); ctx.lineTo(bx + 3, FY(ym - 2 * SIGMA)); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    ctx.fillText("1", fan.x - 5, FY(1) + 4); ctx.fillText("0", fan.x - 5, FY(0) + 4);
    ctx.textAlign = "start";
    ctx.fillStyle = C.ink;
    const fl = "e^(−kt)  ±2σ";
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(fan.x + fan.w - ctx.measureText(fl).width - 12, fan.y + 1, ctx.measureText(fl).width + 11, 18);
    ctx.fillStyle = C.ink; ctx.textAlign = "end"; ctx.fillText(fl, fan.x + fan.w - 6, fan.y + 14); ctx.textAlign = "start";
    // EIG 곡선
    const emax = Math.max(...E.map((e) => e.eig)) * 1.35;
    const EY = (v: number) => eg.y + eg.h - (v / emax) * eg.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(eg.x, eg.y, eg.w, eg.h);
    let ib = 0; E.forEach((e, i) => { if (e.eig > E[ib].eig) ib = i; });
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(TX(eg, TS[ib]), eg.y); ctx.lineTo(TX(eg, TS[ib]), eg.y + eg.h); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(TX(eg, 0), EY(0)); E.forEach((e, i) => ctx.lineTo(TX(eg, TS[i]), EY(e.eig))); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(TX(eg, t), EY(E[ti].eig), 5, 0, 7); ctx.fill();
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const v of [0, 1, 2, 3, 4]) ctx.fillText(String(v), TX(eg, v), eg.y + eg.h + 13);
    ctx.textAlign = "right";
    ctx.fillText("5 s", TX(eg, 5), eg.y + eg.h + 13);
    for (const v of [0, 1]) if (v <= emax) ctx.fillText(String(v), eg.x - 5, EY(v) + 4);
    ctx.textAlign = "start";
    ctx.fillStyle = C.ink; ctx.fillText("EIG (nat)", eg.x + 6, eg.y + 14);
    // 사후분포(log k 축)
    const z0 = Math.log(kmed) - 2.2, z1 = Math.log(kmed) + 2.2;
    const PX = (lk: number) => po.x + ((lk - z0) / (z1 - z0)) * po.w;
    const post = posterior(kmed, t);
    const prior = ZS.map((z) => Math.exp(-0.5 * z * z) / (S_LOG * Math.sqrt(2 * Math.PI)));
    const pmax = Math.max(...post, ...prior) * 1.1;
    const PY = (v: number) => po.y + po.h - (v / pmax) * (po.h - 20);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(po.x, po.y, po.w, po.h);
    ctx.save(); ctx.beginPath(); ctx.rect(po.x, po.y, po.w, po.h); ctx.clip();
    const lk = ZS.map((z) => Math.log(kmed) + S_LOG * z);
    ctx.fillStyle = C.fill;
    ctx.beginPath(); ctx.moveTo(PX(lk[0]), PY(0)); lk.forEach((v, j) => ctx.lineTo(PX(v), PY(post[j]))); ctx.lineTo(PX(lk[NZ - 1]), PY(0)); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); lk.forEach((v, j) => (j ? ctx.lineTo(PX(v), PY(post[j])) : ctx.moveTo(PX(v), PY(post[j])))); ctx.stroke();
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]);
    ctx.beginPath(); lk.forEach((v, j) => (j ? ctx.lineTo(PX(v), PY(prior[j])) : ctx.moveTo(PX(v), PY(prior[j])))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PX(Math.log(kmed)), po.y + 18); ctx.lineTo(PX(Math.log(kmed)), po.y + po.h); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    for (const m of [0.25, 1, 4]) { const v = kmed * m; ctx.fillText(String(v), PX(Math.log(v)), po.y + po.h + 13); }
    ctx.textAlign = "start";
    ctx.fillStyle = C.ink; ctx.fillText("k (1/s, 로그 축)", po.x + 6, po.y + 14);
  }

  function update() {
    const E = eigs(), kmed = PRIORS[key].kmed;
    let ib = 0; E.forEach((e, i) => { if (e.eig > E[ib].eig) ib = i; });
    ro.set("eig", `EIG(${TS[ti].toFixed(2)} s) = ${E[ti].eig.toFixed(3)} nat`, ti === ib ? "ok" : "");
    ro.set("best", `최대 ${E[ib].eig.toFixed(3)} nat @ ${TS[ib].toFixed(2)} s`);
    ro.set("sd", `기대 사후 sd(k) ${E[ti].esd.toFixed(3)} / 사전 ${priorSd(kmed).toFixed(3)}`);
    draw();
  }

  const pt = param({ label: "t", min: 0.05, max: 5, step: 0.05, value: TS[ti], fmt: (v) => `${v.toFixed(2)} s`, aria: "측정 시각 t (초)", onInput: (v) => { ti = Math.round(v / 0.05) - 1; update(); } });
  const pre = presets<Key>((Object.keys(PRIORS) as Key[]).map((k) => ({ key: k, label: PRIORS[k].label })), (k) => { key = k; update(); }, key);
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
