// VAE posterior collapse (data/widgets.toml: vae-collapse, 19.3장 "문제와 대응").
// 본문의 1차원 선형 VAE: 데이터 x ~ N(0, v), decoder N(wz, σ²), encoder N(ax, s²), KL 가중치 β.
// encoder를 닫힌 꼴로 최적화하면 a = w/(w² + βσ²), s² = βσ²/(w² + βσ²)이고, 데이터 평균한 목적함수는
//   재구성 = −½ log(2πσ²) − (v(1 − wa)² + w²s²)/(2σ²),  KL = ½(a²v + s² − 1 − log s²),  J(w) = 재구성 − β·KL.
// 최적 w는 w*² = max(v − βσ², 0)이고, βσ² ≥ v이면 w* = 0(posterior collapse)이다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

const V = 1;
export function terms(w: number, beta: number, sig2: number) {
  const a = w / (w * w + beta * sig2), s2 = (beta * sig2) / (w * w + beta * sig2);
  const rec = -0.5 * Math.log(2 * Math.PI * sig2) - (V * (1 - w * a) ** 2 + w * w * s2) / (2 * sig2);
  const kl = 0.5 * (a * a * V + s2 - 1 - Math.log(s2));
  return { a, s2, rec, kl, obj: rec - beta * kl };
}
/** 본문 셀과 같은 격자 탐색(w ∈ [0, 2], 20001점) */
export function gridBest(beta: number, sig2: number) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i <= 20000; i++) { const v = terms((2 * i) / 20000, beta, sig2).obj; if (v > bv) { bv = v; bi = i; } }
  return (2 * bi) / 20000;
}
export const wStar = (beta: number, sig2: number) => Math.sqrt(Math.max(V - beta * sig2, 0));

const BETAS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12];
type Case = "c1" | "c2" | "c3" | "c4" | "c5";
const CASES: Record<Case, { beta: number; sig2: number }> = {
  c1: { beta: 1, sig2: 0.1 }, c2: { beta: 1, sig2: 0.5 }, c3: { beta: 1, sig2: 1.2 }, c4: { beta: 4, sig2: 0.1 }, c5: { beta: 12, sig2: 0.1 },
};

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 380;
  const f = frame(root, "VAE posterior collapse", "manipulate", narrow ? 0.95 : 0.5,
    "관측 x마다의 encoder 분포 q(z|x)와 사전분포, 그리고 decoder 가중치 w에 따른 재구성 이득·KL 비용·목적함수 곡선");
  let bi = BETAS.indexOf(1), sig2 = 0.5;
  const beta = () => BETAS[bi];

  function layout() {
    const W = f.width, H = f.height;
    if (W < 380) { const h = (H - 100) / 2; return [{ x: 30, y: 40, w: W - 48, h }, { x: 30, y: h + 96, w: W - 48, h: h - 14 }]; }
    const gap = W < 560 ? 44 : 70, w = (W - 30 - gap - 18) / 2;
    return [{ x: 24, y: 40, w, h: H - 68 }, { x: 24 + w + gap, y: 40, w, h: H - 68 }];
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const [g, c] = layout();
    const ws = wStar(beta(), sig2);
    const T = terms(ws, beta(), sig2);
    // 왼쪽: q(z|x) = N(a x, s²), x = ±0.5, ±1.5 (√v 단위), 사전분포 N(0, 1)
    const Z0 = -3.5, Z1 = 3.5;
    const s = Math.sqrt(T.s2);
    const pk = Math.max(1 / Math.sqrt(2 * Math.PI), 1 / (s * Math.sqrt(2 * Math.PI)));
    const X = (z: number) => g.x + ((z - Z0) / (Z1 - Z0)) * g.w;
    const Y = (p: number) => g.y + g.h - (p / (pk * 1.12)) * (g.h - 16);
    const gauss = (z: number, m: number, sd: number) => Math.exp(-0.5 * ((z - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(g.x, g.y + g.h); ctx.lineTo(g.x + g.w, g.y + g.h); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let k = -3; k <= 3; k++) { const t = String(k); ctx.fillText(t, X(k) - ctx.measureText(t).width / 2, g.y + g.h + 13); }
    ctx.fillText("z", g.x + g.w - 6, g.y + g.h - 6);
    const curve = (m: number, sd: number) => { ctx.beginPath(); for (let i = 0; i <= 200; i++) { const z = Z0 + ((Z1 - Z0) * i) / 200; const y = Y(gauss(z, m, sd)); if (i) ctx.lineTo(X(z), y); else ctx.moveTo(X(z), y); } ctx.stroke(); };
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); curve(0, 1); ctx.setLineDash([]);
    const xsObs = [-1.5, -0.5, 0.5, 1.5];
    xsObs.forEach((x, i) => {
      ctx.strokeStyle = i === 3 ? C.edge : alpha(C.edge, 0.35 + 0.15 * i); ctx.lineWidth = 2.2;
      curve(T.a * x, s);
    });
    ctx.lineWidth = 1;
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.edge; ctx.fillText("q(z | x), x = ±0.5, ±1.5", g.x, g.y - 22);
    ctx.fillStyle = C.mid; ctx.fillText("사전분포 N(0, 1) (점선)", g.x, g.y - 6);

    // 오른쪽: w에 따른 재구성 이득, β·KL 비용, 목적함수 J(w) − J(0)
    const W1 = 1.6, n = 160;
    const t0 = terms(0, beta(), sig2);
    const gain: number[] = [], cost: number[] = [], net: number[] = [];
    for (let i = 0; i <= n; i++) { const t = terms((W1 * i) / n, beta(), sig2); gain.push(t.rec - t0.rec); cost.push(beta() * t.kl); net.push(t.obj - t0.obj); }
    const hi = Math.max(1, ...gain.slice(0, Math.round((n * 1.2) / W1))), lo = Math.min(-0.5, ...net);
    const CX = (w: number) => c.x + (w / W1) * c.w;
    const CY = (v: number) => c.y + c.h - ((Math.min(v, hi) - lo) / (hi - lo)) * c.h;
    ctx.strokeStyle = C.faint; ctx.strokeRect(c.x, c.y, c.w, c.h);
    ctx.strokeStyle = C.grid2; ctx.beginPath(); ctx.moveTo(c.x, CY(0)); ctx.lineTo(c.x + c.w, CY(0)); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const w of [0, 0.5, 1, 1.5]) { const t = w.toFixed(1); ctx.fillText(t, CX(w) - ctx.measureText(t).width / 2, c.y + c.h + 13); }
    ctx.fillText("0", c.x - 12, CY(0) + 4);
    ctx.fillText("|w|", c.x + c.w - 18, c.y + c.h - 6);
    ctx.save(); ctx.beginPath(); ctx.rect(c.x, c.y, c.w, c.h); ctx.clip();
    const plot = (arr: number[], col: string, lw: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath();
      arr.forEach((v, i) => { const p = [CX((W1 * i) / n), CY(v)] as const; if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
      ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
    };
    plot(gain, C.ink2, 2);
    plot(cost, C.mid, 1.8, [5, 4]);
    plot(net, C.edge, 2.6);
    ctx.restore();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(CX(ws), CY(T.obj - t0.obj), 5.5, 0, 7); ctx.fill();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    let lx = c.x; // 범례는 그림 위 한 줄
    for (const [t, col] of [["재구성 이득", C.ink2], ["β·KL", C.mid], ["J(w) − J(0)", C.edge]] as const) {
      ctx.fillStyle = col; ctx.fillText(t, lx, c.y - 6); lx += ctx.measureText(t).width + 12;
    }
  }

  function update() {
    const b = beta(), ws = wStar(b, sig2), T = terms(ws, b, sig2), wg = gridBest(b, sig2);
    const collapse = b * sig2 >= V;
    ro.set("th", `βσ² ${(b * sig2).toFixed(2)} ${collapse ? "≥" : "<"} v: ${collapse ? "collapse" : "잠재변수 사용"}`, collapse ? "hot" : "ok");
    ro.set("w", `최적 |w| ${wg.toFixed(3)} (공식 ${ws.toFixed(3)}) · KL ${Math.max(T.kl, 0).toFixed(3)}`);
    ro.set("q", `q(z|x) = N(${T.a.toFixed(2)}x, ${Math.sqrt(T.s2).toFixed(2)}²)`);
    draw();
  }

  const cs = presets((Object.keys(CASES) as Case[]).map((k) => ({ key: k, label: `β=${CASES[k].beta}, σ²=${CASES[k].sig2}` })), (k) => {
    bi = BETAS.indexOf(CASES[k].beta); sig2 = CASES[k].sig2; pb.set(bi); psg.set(sig2); update();
  }, "c2");
  cs.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 가로로 넘긴다
  const pb = param({ label: "β", min: 0, max: BETAS.length - 1, step: 1, value: bi, fmt: (i) => String(BETAS[i]), aria: "KL 항 가중치 β", onInput: (i) => { bi = i; cs.select(null); update(); } });
  const psg = param({ label: "σ²", min: 0.05, max: 2, step: 0.05, value: sig2, fmt: (v) => v.toFixed(2), aria: "decoder 잡음 분산 σ²", onInput: (v) => { sig2 = v; cs.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pb.row, psg.row);
  const ro = readouts();
  f.panel.append(cs.row, grid, ro.row);
  f.onResize(draw);
  update();
}
