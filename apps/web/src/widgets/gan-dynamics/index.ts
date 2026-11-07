// GAN 학습 동역학 (data/widgets.toml: gan-dynamics, 19.4장 "GAN의 어려움").
// 본문 셀과 같은 장난감: 데이터 ½N(−2, 0.5²) + ½N(2, 0.5²), 가우시안 생성자 x = μ + s·ε(μ와 log s를 학습),
// RBF 특징 17개(중심 −4…4, 폭 0.5) 위의 로지스틱 판별자. 한 단계마다 판별자 2번(log D(real) + log(1 − D(fake)) 상승),
// 생성자 1번(non-saturating, log D(fake) 상승). 판별자 학습률 η_D = 1(본문과 같음)로 두고 η_G를 학습률 비로 바꾼다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

export const CENTERS = Array.from({ length: 17 }, (_, i) => -4 + 0.5 * i);
const B = 128, STEPS = 600;
const sig = (z: number) => 1 / (1 + Math.exp(-z));
const phi = (x: number, j: number) => Math.exp(-((x - CENTERS[j]) ** 2) / (2 * 0.25));
export const logit = (x: number, w: ArrayLike<number>, b: number) => { let s = b; for (let j = 0; j < 17; j++) s += w[j] * phi(x, j); return s; };

export interface Snap { mu: number; s: number; w: Float64Array; b: number }
export interface Params { mu0: number; s0: number; ratio: number; seed: number; kD?: number; record?: number[] }

/** 600단계 학습 궤적. hist[t]는 t번째 갱신 뒤의 상태(hist[0]은 출발점). record가 있으면 쓴 난수를 순서대로 담는다. */
export function simulate(p: Params): Snap[] {
  const r = rng(p.seed * 7907 + 3);
  const u = () => { const v = r.u(); p.record?.push(v); return v; };
  const nrm = () => { const v = r.normal(); p.record?.push(v); return v; };
  const kD = p.kD ?? 2, etaD = 1, etaG = p.ratio * etaD;
  const w = new Float64Array(17);
  let b = 0, mu = p.mu0, s = p.s0;
  const hist: Snap[] = [{ mu, s, w: new Float64Array(w), b }];
  const xr = new Float64Array(B), eps = new Float64Array(B), xf = new Float64Array(B);
  const Fr = new Float64Array(B * 17), Ff = new Float64Array(B * 17);
  for (let t = 0; t < STEPS; t++) {
    for (let i = 0; i < B; i++) xr[i] = u() < 0.5 ? -2 : 2;
    for (let i = 0; i < B; i++) xr[i] += 0.5 * nrm();
    for (let i = 0; i < B; i++) { eps[i] = nrm(); xf[i] = mu + s * eps[i]; }
    for (let i = 0; i < B; i++) for (let j = 0; j < 17; j++) { Fr[i * 17 + j] = phi(xr[i], j); Ff[i * 17 + j] = phi(xf[i], j); }
    for (let k = 0; k < kD; k++) {
      const gw = new Float64Array(17);
      let gb = 0;
      for (let i = 0; i < B; i++) {
        let zr = b, zf = b;
        for (let j = 0; j < 17; j++) { zr += w[j] * Fr[i * 17 + j]; zf += w[j] * Ff[i * 17 + j]; }
        const pr = sig(zr), pf = sig(zf);
        for (let j = 0; j < 17; j++) gw[j] += ((1 - pr) * Fr[i * 17 + j] - pf * Ff[i * 17 + j]) / B;
        gb += (1 - pr - pf) / B;
      }
      for (let j = 0; j < 17; j++) w[j] += etaD * gw[j];
      b += etaD * gb;
    }
    let gmu = 0, gs = 0;
    for (let i = 0; i < B; i++) {
      let z = b, dz = 0;
      for (let j = 0; j < 17; j++) { const f = Ff[i * 17 + j]; z += w[j] * f; dz += w[j] * f * (-(xf[i] - CENTERS[j]) / 0.25); }
      const pf = sig(z);
      gmu += ((1 - pf) * dz) / B;
      gs += ((1 - pf) * dz * eps[i]) / B;
    }
    mu += etaG * gmu;
    s *= Math.exp(etaG * gs * s);
    hist.push({ mu, s, w: new Float64Array(w), b });
  }
  return hist;
}

const lnN = (x: number, m: number, sd: number) => -0.5 * ((x - m) / sd) ** 2 - Math.log(sd) - 0.5 * Math.log(2 * Math.PI);
const lnData = (x: number) => { const a = lnN(x, -2, 0.5), c = lnN(x, 2, 0.5); const m = Math.max(a, c); return m + Math.log(0.5 * Math.exp(a - m) + 0.5 * Math.exp(c - m)); };
const lae = (a: number, c: number) => { const m = Math.max(a, c); return m + Math.log(Math.exp(a - m) + Math.exp(c - m)); };
/** JS(p_data ‖ N(μ, s²)). 두 항을 각 분포의 지지 영역 격자에서 따로 적분한다. */
export function jsDiv(mu: number, s: number) {
  let a = 0, c = 0;
  const n = 2000;
  for (let i = 0; i <= n; i++) { // E_data[log p/m]
    const x = -5 + (10 * i) / n, lp = lnData(x), lq = lnN(x, mu, s), lm = lae(lp, lq) - Math.LN2;
    a += Math.exp(lp) * (lp - lm) * (10 / n) * (i === 0 || i === n ? 0.5 : 1);
  }
  for (let i = 0; i <= n; i++) { // E_g[log q/m]
    const x = mu - 8 * s + (16 * s * i) / n, lp = lnData(x), lq = lnN(x, mu, s), lm = lae(lp, lq) - Math.LN2;
    c += Math.exp(lq) * (lq - lm) * ((16 * s) / n) * (i === 0 || i === n ? 0.5 : 1);
  }
  return 0.5 * a + 0.5 * c;
}

type Scene = "text" | "wide" | "fast";
const SCENES: Record<Scene, { label: string; s0: number; ratio: number }> = {
  text: { label: "본문 설정", s0: 0.3, ratio: 0.05 },
  wide: { label: "넓게 출발", s0: 1.0, ratio: 0.05 },
  fast: { label: "생성자 과속", s0: 0.3, ratio: 2 },
};
const RATIOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3];

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 560;
  const f = frame(root, "GAN 학습 동역학", "manipulate", narrow ? 0.72 : 0.46,
    "데이터 밀도, 생성자 밀도, 판별자 출력 D(x)와 학습 단계에 따른 생성자 위치의 궤적");
  let s0 = 0.3, ri = RATIOS.indexOf(0.05), step = STEPS;
  const seed = 1;
  const MU0 = -0.3;
  let hist: Snap[] = [];
  const X0 = -8, X1 = 8;

  function layout() {
    const W = f.width, H = f.height;
    const top = { x: 34, y: 26, w: W - 68, h: H * 0.58 - 34 };
    const bot = { x: 34, y: H * 0.58 + 22, w: W - 68, h: H * 0.42 - 44 };
    return { top, bot };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const { top: g, bot } = layout();
    const S = hist[step];
    const X = (x: number) => g.x + ((x - X0) / (X1 - X0)) * g.w;
    const YMAX = 0.85;
    const Yp = (v: number) => g.y + g.h - (Math.min(v, YMAX) / YMAX) * g.h;
    const Yd = (v: number) => g.y + g.h - v * g.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(g.x, g.y + g.h); ctx.lineTo(g.x + g.w, g.y + g.h); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (let k = -8; k <= 8; k += 2) { const t = String(k); ctx.fillText(t, X(k) - ctx.measureText(t).width / 2, g.y + g.h + 13); }
    // D = 1/2 기준선
    ctx.strokeStyle = C.grid2; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(g.x, Yd(0.5)); ctx.lineTo(g.x + g.w, Yd(0.5)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("½", g.x + g.w + 4, Yd(0.5) + 4);
    ctx.fillText("1", g.x + g.w + 4, Yd(1) + 8);
    const M = 400;
    // 데이터 밀도(면)
    ctx.fillStyle = C.grid2;
    ctx.beginPath(); ctx.moveTo(X(X0), Yp(0));
    for (let i = 0; i <= M; i++) { const x = X0 + ((X1 - X0) * i) / M; ctx.lineTo(X(x), Yp(Math.exp(lnData(x)))); }
    ctx.lineTo(X(X1), Yp(0)); ctx.closePath(); ctx.fill();
    // 판별자 D(x)
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= M; i++) { const x = X0 + ((X1 - X0) * i) / M; const y = Yd(sig(logit(x, S.w, S.b))); if (i) ctx.lineTo(X(x), y); else ctx.moveTo(X(x), y); }
    ctx.stroke();
    // 생성자 밀도: 좁으면 격자를 생성자 둘레로 촘촘하게
    ctx.save(); ctx.beginPath(); ctx.rect(g.x, g.y - 4, g.w, g.h + 4); ctx.clip();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
    ctx.beginPath();
    const lo = Math.max(X0, S.mu - 6 * S.s), hi = Math.min(X1, S.mu + 6 * S.s);
    ctx.moveTo(X(X0), Yp(0)); ctx.lineTo(X(lo), Yp(0));
    for (let i = 0; i <= M; i++) { const x = lo + ((hi - lo) * i) / M; ctx.lineTo(X(x), Yp(Math.exp(lnN(x, S.mu, S.s)))); }
    ctx.lineTo(X(X1), Yp(0));
    ctx.stroke(); ctx.restore();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    let lx = g.x;
    for (const [t, col] of [["데이터", C.mid], ["생성자", C.edge], ["판별자 D(x)", C.ink2]] as const) {
      ctx.fillStyle = col; ctx.fillText(t, lx, 14); lx += ctx.measureText(t).width + 12;
    }

    // 아래: 단계에 따른 μ ± s 궤적
    const TX = (t: number) => bot.x + (t / STEPS) * bot.w;
    const TY = (x: number) => bot.y + bot.h - ((Math.max(X0, Math.min(X1, x)) - X0) / (X1 - X0)) * bot.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(bot.x, bot.y, bot.w, bot.h);
    ctx.strokeStyle = C.grid2; ctx.setLineDash([3, 4]);
    for (const m of [-2, 2]) { ctx.beginPath(); ctx.moveTo(bot.x, TY(m)); ctx.lineTo(bot.x + bot.w, TY(m)); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    if (TY(-2) - TY(2) > 14) { ctx.fillText("−2", bot.x - 22, TY(-2) + 4); ctx.fillText("2", bot.x - 14, TY(2) + 4); }
    else ctx.fillText("±2", bot.x - 24, TY(0) + 4);
    ctx.fillText("단계", bot.x + bot.w - 26, bot.y + bot.h + 13);
    ctx.fillText("0", bot.x, bot.y + bot.h + 13); ctx.fillText(String(STEPS), bot.x + bot.w - 58, bot.y + bot.h + 13);
    ctx.fillStyle = C.fill;
    ctx.beginPath();
    for (let t = 0; t <= STEPS; t++) ctx.lineTo(TX(t), TY(hist[t].mu + hist[t].s));
    for (let t = STEPS; t >= 0; t--) ctx.lineTo(TX(t), TY(hist[t].mu - hist[t].s));
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.edgeSoft; ctx.lineWidth = 1.5;
    ctx.beginPath(); for (let t = 0; t <= STEPS; t++) ctx.lineTo(TX(t), TY(hist[t].mu)); ctx.stroke();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
    ctx.beginPath(); for (let t = 0; t <= step; t++) ctx.lineTo(TX(t), TY(hist[t].mu)); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(TX(step), TY(S.mu), 4.5, 0, 7); ctx.fill();
    ctx.lineWidth = 1;
    ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("μ ± s", bot.x + 6, bot.y + 15);
  }

  function state(S: Snap, js: number) {
    if (js > 0.6) return { t: "데이터 밖에서 멈춤: 기울기 소실", tone: "hot" as const };
    if (S.s > 1.2) return { t: "두 봉우리를 넓게 덮음", tone: "ok" as const };
    if (Math.abs(Math.abs(S.mu) - 2) < 0.4 && S.s < 0.8) return { t: "한 봉우리에 갇힘: mode collapse", tone: "hot" as const };
    return { t: "이동 중", tone: "" as const };
  }

  function update(resim = false) {
    if (resim) hist = simulate({ mu0: MU0, s0, ratio: RATIOS[ri], seed });
    const S = hist[step];
    const js = jsDiv(S.mu, S.s);
    const st = state(S, js);
    const D = (x: number) => sig(logit(x, S.w, S.b)).toFixed(2);
    ro.set("g", `t ${step} · N(${S.mu.toFixed(2)}, ${S.s.toFixed(2)}²) · JS ${js.toFixed(3)}`);
    ro.set("d", `D(−2) ${D(-2)} · D(2) ${D(2)}`);
    ro.set("st", st.t, st.tone);
    draw();
  }

  // 학습 궤적 재계산(약 0.1초)은 슬라이더를 끄는 동안 프레임마다 한 번만
  let queued = false;
  const resim = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; update(true); }); };
  const scene = presets((Object.keys(SCENES) as Scene[]).map((k) => ({ key: k, label: SCENES[k].label })), (k) => {
    s0 = SCENES[k].s0; ri = RATIOS.indexOf(SCENES[k].ratio); step = STEPS;
    ps.set(s0); pr.set(ri); pt.set(step); update(true);
  }, "text");
  scene.row.classList.add("segmented");
  const pr = param({ label: "η_G", min: 0, max: RATIOS.length - 1, step: 1, value: ri, fmt: (i) => String(RATIOS[i]), aria: "생성자 학습률 η_G (판별자 학습률 η_D = 1)", onInput: (i) => { ri = i; scene.select(null); resim(); } });
  const ps = param({ label: "s₀", min: 0.2, max: 2, step: 0.1, value: s0, fmt: (v) => v.toFixed(1), aria: "생성자의 초기 표준편차", onInput: (v) => { s0 = v; scene.select(null); resim(); } });
  const pt = param({ label: "t", min: 0, max: STEPS, step: 5, value: step, fmt: (v) => String(v), aria: "보여 줄 학습 단계", onInput: (v) => { step = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pr.row, ps.row, pt.row);
  const ro = readouts();
  f.panel.append(scene.row, grid, ro.row);
  f.onResize(draw);
  update(true);
}
