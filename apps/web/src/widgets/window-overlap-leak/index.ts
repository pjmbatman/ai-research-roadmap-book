// window 겹침 누수 (data/widgets.toml: window-overlap-leak). 18.7장의 합성 기온 시계열과 분할 규칙을 그대로 쓴다.
// 120일 시간별 기온 = 10 + 0.003t + 5 sin(2π(t − 9)/24) + AR(1) 잡음(0.97, 0.4). lookback L, horizon H = 24, stride로 window를 자른다.
// 무작위 분할: window를 섞어 80%를 학습에. 시간 순서 분할: 정답까지 96일째 이전에 끝나는 window는 학습, 정답이 모두 그 이후인 window는 평가.
// 읽음값: test 정답 구간을 train window가 이미 본 비율(장의 코드와 같은 셈), 1-최근접 이웃(가장 비슷한 train 입력의 미래를 복사)의 test MAE,
// 같은 test window에서 seasonal naive(24시간 전 값)의 MAE. 1-NN은 계산량 때문에 test window를 고르게 최대 150개만 쓴다.
// 아래 그림: test window 하나의 확대(정답 구간, 1-NN 예측 점선)와, 그 정답 구간과 겹치는 train window들(한 줄에 하나, 1-NN이 고른 것은 진하게).
import { frame, param, presets, readouts, el, C, rng } from "../frame";

export const NH = 24 * 120, HZ = 24, CUT = 24 * 96, MAX_TEST = 150;
export const TEMP = (() => {
  const r = rng(77);
  const out = new Float64Array(NH);
  let noise = 0;
  for (let t = 0; t < NH; t++) {
    if (t > 0) noise = 0.97 * noise + 0.4 * r.normal();
    out[t] = 10 + 0.003 * t + 5 * Math.sin((2 * Math.PI * (t - 9)) / 24) + noise;
  }
  return out;
})();
export type Split = "random" | "time";

export function evaluate(L: number, stride: number, split: Split) {
  const starts: number[] = [];
  for (let s = 0; s + L + HZ <= NH; s += stride) starts.push(s);
  let tr: number[] = [], te: number[] = [];
  if (split === "random") {
    const r = rng(2809 + L * 7 + stride);
    const idx = starts.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const ntr = Math.floor(0.8 * idx.length);
    tr = idx.slice(0, ntr).map((i) => starts[i]); te = idx.slice(ntr).map((i) => starts[i]);
  } else {
    tr = starts.filter((s) => s + L + HZ <= CUT); te = starts.filter((s) => s + L >= CUT);
  }
  tr.sort((a, b) => a - b); te.sort((a, b) => a - b);
  // train window가 본 시점
  const seen = new Uint8Array(NH);
  for (const s of tr) for (let t = s; t < s + L + HZ; t++) seen[t] = 1;
  let leaked = 0;
  for (const s of te) { let all = true; for (let t = s + L; t < s + L + HZ; t++) if (!seen[t]) { all = false; break; } if (all) leaked++; }
  // 1-NN과 seasonal naive (test window를 고르게 최대 MAX_TEST개)
  const pick = te.length <= MAX_TEST ? te : Array.from({ length: MAX_TEST }, (_, i) => te[Math.floor(((i + 0.5) * te.length) / MAX_TEST)]);
  const nn = (s: number) => {
    let best = -1, bd = Infinity;
    for (const u of tr) {
      let d = 0;
      for (let k = 0; k < L && d < bd; k++) { const v = TEMP[s + k] - TEMP[u + k]; d += v * v; }
      if (d < bd) { bd = d; best = u; }
    }
    return best;
  };
  let maeNN = 0, maeSN = 0, nSN = 0;
  for (const s of pick) {
    const u = nn(s);
    for (let h = 0; h < HZ; h++) {
      const y = TEMP[s + L + h];
      maeNN += Math.abs(TEMP[u + L + h] - y);
      if (s + L + h - 24 >= 0) { maeSN += Math.abs(TEMP[s + L + h - 24] - y); nSN++; }   // 24시간 전 값이 있는 시점만
    }
  }
  const ex = pick.length ? pick[Math.floor(pick.length / 2)] : -1;
  return { starts, tr, te, seen, leakFrac: te.length ? leaked / te.length : 0, maeNN: maeNN / (pick.length * HZ), maeSN: maeSN / nSN, ex, exNN: ex >= 0 ? nn(ex) : -1 };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "window 겹침 누수", "manipulate", narrow ? 1.1 : 0.62, "120일 기온 시계열과 train·test window가 덮는 시점, 그리고 test window 하나와 1-최근접 이웃이 고른 train window");
  let L = 48, stride = 1, split: Split = "random";
  let res = evaluate(L, stride, split);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pad = 14, lab = narrow ? 64 : 84;
    const x0 = pad + lab, x1 = W - pad;
    let tmin = Infinity, tmax = -Infinity; TEMP.forEach((v) => { if (v < tmin) tmin = v; if (v > tmax) tmax = v; });
    // 위: 전체 시계열
    const a0 = 10, a1 = Math.round(H * 0.26);
    const TX = (t: number) => x0 + (t / NH) * (x1 - x0);
    const TY = (v: number) => a1 - ((v - tmin) / (tmax - tmin)) * (a1 - a0);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 0.8;
    ctx.beginPath(); for (let t = 0; t < NH; t += 2) (t ? ctx.lineTo(TX(t), TY(TEMP[t])) : ctx.moveTo(TX(t), TY(TEMP[t]))); ctx.stroke();
    if (split === "time") { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(TX(CUT), a0 - 4); ctx.lineTo(TX(CUT), a1 + 44); ctx.stroke(); ctx.setLineDash([]); }
    // 띠: train window가 본 시점 / test 정답 시점
    const r1 = a1 + 8, r2 = a1 + 26, rh = 12;
    const tgt = new Uint8Array(NH);
    for (const s of res.te) for (let t = s + L; t < s + L + HZ; t++) tgt[t] = 1;
    const band = (y: number, on: Uint8Array, col: string) => {
      ctx.fillStyle = col;
      let t = 0;
      while (t < NH) { if (!on[t]) { t++; continue; } let e = t; while (e < NH && on[e]) e++; ctx.fillRect(TX(t), y, Math.max(1, TX(e) - TX(t)), rh); t = e; }
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y, x1 - x0, rh);
    };
    band(r1, res.seen, C.ink2);
    band(r2, tgt, C.edge);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.ink; ctx.fillText(narrow ? "train 시점" : "train이 본 시점", pad, r1 + 10);
    ctx.fillStyle = C.edge; ctx.fillText(narrow ? "test 정답" : "test 정답 시점", pad, r2 + 10);
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const d of narrow ? [0, 30, 60, 96] : [0, 30, 60, 96, 120]) { const s = `${d}일`; ctx.fillText(s, Math.min(x1 - ctx.measureText(s).width, Math.max(x0, TX(d * 24) - ctx.measureText(s).width / 2)), r2 + rh + 13); }
    // 아래: test window 하나의 확대
    if (res.ex < 0) return;
    const s = res.ex, u = res.exNN;
    // 확대 구간: 이 test window의 정답 구간과 겹칠 수 있는 모든 window의 범위 (s − H, s + 2(L + H))
    const z0 = Math.max(0, s - HZ), z1 = Math.min(NH, s + 2 * (L + HZ));
    const bandH = narrow ? 40 : 46;
    const b0 = r2 + rh + 34, b1 = H - 12 - 16 - bandH - 8;
    const ZX = (t: number) => x0 + ((t - z0) / (z1 - z0)) * (x1 - x0);
    let zmin = Infinity, zmax = -Infinity;
    for (let t = z0; t < z1; t++) { zmin = Math.min(zmin, TEMP[t]); zmax = Math.max(zmax, TEMP[t]); }
    for (let h = 0; h < HZ; h++) { zmin = Math.min(zmin, TEMP[u + L + h]); zmax = Math.max(zmax, TEMP[u + L + h]); }
    const ZY = (v: number) => b1 - ((v - zmin) / (zmax - zmin || 1)) * (b1 - b0);
    // test window의 정답 구간
    ctx.fillStyle = C.fill;
    ctx.fillRect(ZX(s + L), b0, ZX(s + L + HZ) - ZX(s + L), b1 - b0);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); for (let t = z0; t < z1; t++) (t > z0 ? ctx.lineTo(ZX(t), ZY(TEMP[t])) : ctx.moveTo(ZX(t), ZY(TEMP[t]))); ctx.stroke();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2;
    ctx.beginPath(); for (let h = 0; h < HZ; h++) (h ? ctx.lineTo(ZX(s + L + h), ZY(TEMP[s + L + h])) : ctx.moveTo(ZX(s + L + h), ZY(TEMP[s + L + h]))); ctx.stroke();
    // 1-NN 예측: 고른 train window의 미래를 그대로 복사
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2; ctx.setLineDash([5, 3]);
    ctx.beginPath(); for (let h = 0; h < HZ; h++) { const v = ZY(TEMP[u + L + h]); h ? ctx.lineTo(ZX(s + L + h), v) : ctx.moveTo(ZX(s + L + h), v); } ctx.stroke();
    ctx.setLineDash([]);
    // window 막대: [입력 L(옅게) | 정답 H(진하게)]
    const bar = (y: number, h: number, st: number, col: string) => {
      const xa = Math.max(x0, ZX(st)), xb = Math.min(x1, ZX(st + L + HZ)), xm = Math.min(Math.max(ZX(st + L), xa), xb);
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.4; ctx.fillRect(xa, y, xm - xa, h);
      ctx.globalAlpha = 1; ctx.fillRect(xm, y, xb - xm, h);
    };
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const by = b1 + 10;
    bar(by, 10, s, C.edge);
    ctx.fillStyle = C.edge; ctx.fillText("test window", pad, by + 9);
    // 이 test window의 정답 구간과 겹치는 train window들(한 줄에 하나)
    const over = res.tr.filter((v) => v > s - HZ && v < s + L + HZ);
    const y0 = by + 16, rowH = over.length ? Math.min(6, bandH / over.length) : 0;
    over.forEach((v, i) => bar(y0 + i * rowH, Math.max(1, rowH - (rowH > 2 ? 1 : 0)), v, C.ink2));
    const ui = over.indexOf(u);
    if (ui >= 0) bar(y0 + ui * rowH + rowH / 2 - 1.5, 3, u, C.ink);   // 1-NN이 고른 train window
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, bandH);
    ctx.fillStyle = C.ink; ctx.fillText(narrow ? `겹친 train ${over.length}` : `겹친 train ${over.length}개`, pad, y0 + 12);
  }

  function update() {
    res = evaluate(L, stride, split);
    ro.set("leak", `train이 본 test 정답 ${(res.leakFrac * 100).toFixed(1)}%`, res.leakFrac > 0.01 ? "hot" : "ok");
    ro.set("nn", `1-NN MAE ${res.maeNN.toFixed(2)} °C`);
    ro.set("sn", `seasonal naive ${res.maeSN.toFixed(2)} °C`);
    const dt = res.ex >= 0 ? res.exNN - res.ex : 0;
    ro.set("dt", `이웃 Δt ${dt > 0 ? "+" : dt < 0 ? "−" : ""}${Math.abs(dt)}h`);
    draw();
  }

  const splitBtns = presets<Split>([{ key: "random", label: "무작위 분할" }, { key: "time", label: "시간 순서 분할" }], (k) => { split = k; update(); }, split);
  splitBtns.row.classList.add("segmented");
  let raf = 0;
  const later = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(update); };
  const pl = param({ label: "L", min: 6, max: 96, step: 6, value: L, fmt: (v) => `${v}h`, aria: "lookback window 길이 L (시간)", onInput: (v) => { L = v; later(); } });
  const ps = param({ label: "stride", min: 1, max: 96, step: 1, value: stride, fmt: (v) => `${v}h`, aria: "window 사이 간격 stride (시간)", onInput: (v) => { stride = v; later(); } });
  ps.row.style.gridTemplateColumns = "3.8em minmax(0, 1fr) 3.6em";
  pl.row.style.gridTemplateColumns = "3.8em minmax(0, 1fr) 3.6em";
  const grid = el("div", "param-grid");
  grid.append(pl.row, ps.row);
  const ro = readouts();
  f.panel.append(splitBtns.row, grid, ro.row);
  f.onResize(draw);
  update();
}
