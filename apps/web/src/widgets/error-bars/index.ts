// SD·SE·CI 오차막대 (data/widgets.toml: error-bars).
// 23.6장 ablation 표의 구조를 따른다: seed마다 모든 모델이 함께 흔들리는 공통 효과(표준편차 0.04)와 모델별 잡음(0.02),
// 참값 RMSE는 B 전체 1.00, − 물리 손실 1.12, − attention 1.01, A 확대 1.09. seed 30개를 미리 뽑아 두고 앞의 n개를 쓴다.
// 왼쪽: 두 모델의 seed별 점(같은 seed는 선으로 이음)과 고른 종류의 오차막대. 오른쪽: 같은 seed끼리 짝지은 차이와 95% t 구간.
import { frame, param, presets, readouts, el, C, rng } from "../frame";

// t_{0.975, df}, df = 1..29 (scipy.stats.t.ppf로 확인)
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045];
const NMAX = 30;
const MODELS = {
  phys: { label: "− 물리 손실", truth: 1.12 },
  attn: { label: "− attention", truth: 1.01 },
  bigA: { label: "A 확대", truth: 1.09 },
} as const;
type MK = keyof typeof MODELS;
type Bar = "sd" | "se" | "ci";

export function pool(seed: number) {
  const r = rng(seed);
  const fx = Array.from({ length: NMAX }, () => 0.04 * r.normal());
  const full = fx.map((v) => 1.0 + v + 0.02 * r.normal());
  const other = Object.fromEntries((Object.keys(MODELS) as MK[]).map((k) => [k, fx.map((v) => MODELS[k].truth + v + 0.02 * r.normal())])) as Record<MK, number[]>;
  return { full, other };
}
export function stat(v: number[]) {
  const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const se = sd / Math.sqrt(n);
  return { mean, sd, se, ci: T975[n - 2] * se };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "SD·SE·CI 오차막대", "compare", narrow ? 0.95 : 0.56, "두 모델의 seed별 RMSE와 오차막대, 그리고 같은 seed끼리 짝지은 차이와 95% 구간");
  let model: MK = (attrs.model as MK) in MODELS ? (attrs.model as MK) : "phys";
  let bar: Bar = "sd", n = 5, seed = 7;
  let data = pool(seed);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const a = data.full.slice(0, n), b = data.other[model].slice(0, n);
    const sa = stat(a), sb = stat(b);
    const d = b.map((v, i) => v - a[i]), sd = stat(d);
    const ha = sa[bar], hb = sb[bar];
    // 왼쪽 패널: RMSE 축
    const lw = W * (narrow ? 0.58 : 0.6), top = 40, bottom = H - 34;
    const all = [...data.full, ...data.other[model]];
    const lo = Math.min(...all, sa.mean - ha, sb.mean - hb) - 0.02, hi = Math.max(...all, sa.mean + ha, sb.mean + hb) + 0.02;
    const Y = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
    const axL = narrow ? 40 : 48;
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText(`RMSE, seed ${n}개 · 막대 ${bar === "sd" ? "SD" : bar === "se" ? "SE" : "95% CI"}`, 6, 16);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(axL, top - 6); ctx.lineTo(axL, bottom); ctx.stroke();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    const step = hi - lo > 0.4 ? 0.1 : 0.05;
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      const s = t.toFixed(2);
      ctx.fillText(s, axL - 5 - ctx.measureText(s).width, Y(t) + 4);
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(axL, Y(t)); ctx.lineTo(axL + 4, Y(t)); ctx.stroke();
    }
    const cx = [axL + (lw - axL) * 0.3, axL + (lw - axL) * 0.74];
    // 막대 겹침 구간
    const ovLo = Math.max(sa.mean - ha, sb.mean - hb), ovHi = Math.min(sa.mean + ha, sb.mean + hb);
    if (ovHi > ovLo) {
      ctx.fillStyle = C.edgeSoft;
      ctx.fillRect(axL + 1, Y(ovHi), lw - axL - 6, Y(ovLo) - Y(ovHi));
    }
    // seed 짝 선
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.moveTo(cx[0] + 10, Y(a[i])); ctx.lineTo(cx[1] + 10, Y(b[i])); ctx.stroke(); }
    const column = (v: number[], s: ReturnType<typeof stat>, h: number, x: number, col: string, name: string) => {
      ctx.fillStyle = col;
      v.forEach((y) => { ctx.beginPath(); ctx.arc(x + 10, Y(y), 3, 0, 7); ctx.fill(); });
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 6, Y(s.mean + h)); ctx.lineTo(x - 6, Y(s.mean - h));
      ctx.moveTo(x - 12, Y(s.mean + h)); ctx.lineTo(x, Y(s.mean + h));
      ctx.moveTo(x - 12, Y(s.mean - h)); ctx.lineTo(x, Y(s.mean - h));
      ctx.stroke();
      ctx.fillStyle = C.ink;
      ctx.beginPath(); ctx.arc(x - 6, Y(s.mean), 4.5, 0, 7); ctx.fill();
      ctx.font = "12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
      ctx.fillText(name, x - ctx.measureText(name).width / 2, bottom + 18);
    };
    column(a, sa, ha, cx[0], C.ink2, "B 전체");
    column(b, sb, hb, cx[1], C.edge, MODELS[model].label);
    // 오른쪽 패널: 짝지은 차이
    const rx0 = lw + (narrow ? 8 : 18), rw = W - rx0 - 8;
    const dmax = Math.max(0.05, ...d.map(Math.abs), Math.abs(sd.mean) + sd.ci) * 1.15;
    const dlo = Math.min(-0.03, -dmax * 0.25, sd.mean - sd.ci - 0.01), dhi = Math.max(dmax, sd.mean + sd.ci + 0.01);
    const Yd = (v: number) => bottom - ((v - dlo) / (dhi - dlo)) * (bottom - top);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText(narrow ? "짝지은 차이" : "짝지은 차이 · 95% CI", rx0, 16);
    ctx.strokeStyle = C.mid; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(rx0 + 40, Yd(0)); ctx.lineTo(rx0 + rw, Yd(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    const dstep = dhi - dlo > 0.3 ? 0.1 : 0.05;
    for (let t = Math.ceil(dlo / dstep) * dstep; t <= dhi; t += dstep) {
      const s = Math.abs(t) < 1e-9 ? "0" : (t > 0 ? "+" : "−") + Math.abs(t).toFixed(2);
      ctx.fillText(s, rx0, Yd(t) + 4);
    }
    const dx = rx0 + 40 + (rw - 40) * 0.45;
    ctx.fillStyle = C.grid2;
    d.forEach((v, i) => { ctx.beginPath(); ctx.arc(dx + 12 + ((i % 5) - 2) * 3, Yd(v), 2.6, 0, 7); ctx.fill(); });
    const excl = sd.mean - sd.ci > 0 || sd.mean + sd.ci < 0;
    ctx.strokeStyle = excl ? C.edge : C.ink; ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(dx - 6, Yd(sd.mean + sd.ci)); ctx.lineTo(dx - 6, Yd(sd.mean - sd.ci));
    ctx.moveTo(dx - 12, Yd(sd.mean + sd.ci)); ctx.lineTo(dx, Yd(sd.mean + sd.ci));
    ctx.moveTo(dx - 12, Yd(sd.mean - sd.ci)); ctx.lineTo(dx, Yd(sd.mean - sd.ci));
    ctx.stroke();
    ctx.fillStyle = excl ? C.edge : C.ink;
    ctx.beginPath(); ctx.arc(dx - 6, Yd(sd.mean), 4.5, 0, 7); ctx.fill();
    ctx.font = "12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    const lab = "비교 − B 전체";
    ctx.fillText(lab, Math.max(rx0, dx - ctx.measureText(lab).width / 2), bottom + 18);
  }

  function update() {
    const a = data.full.slice(0, n), b = data.other[model].slice(0, n);
    const sa = stat(a), sb = stat(b), sd = stat(b.map((v, i) => v - a[i]));
    const ha = sa[bar], hb = sb[bar];
    const overlap = Math.min(sa.mean + ha, sb.mean + hb) > Math.max(sa.mean - ha, sb.mean - hb);
    const name = bar === "sd" ? "SD" : bar === "se" ? "SE" : "95% CI";
    ro.set("a", `B 전체 ${sa.mean.toFixed(3)} ± ${ha.toFixed(3)}`);
    ro.set("b", `${MODELS[model].label} ${sb.mean.toFixed(3)} ± ${hb.toFixed(3)}`);
    ro.set("ov", overlap ? `${name} 막대 겹침` : `${name} 막대 안 겹침`, overlap ? "hot" : "");
    const excl = sd.mean - sd.ci > 0 || sd.mean + sd.ci < 0;
    ro.set("d", `짝지은 차이 ${sd.mean >= 0 ? "+" : "−"}${Math.abs(sd.mean).toFixed(3)} ± ${sd.ci.toFixed(3)} · 0 ${excl ? "밖" : "포함"}`, excl ? "ok" : "");
    draw();
  }

  const barBtns = presets<Bar>([{ key: "sd", label: "SD" }, { key: "se", label: "SE" }, { key: "ci", label: "95% CI" }], (k) => { bar = k; update(); }, bar);
  barBtns.row.classList.add("segmented");
  const modelBtns = presets((Object.keys(MODELS) as MK[]).map((k) => ({ key: k, label: MODELS[k].label })), (k) => { model = k; update(); }, model);
  const pn = param({ label: "seed 수 n", min: 3, max: NMAX, step: 1, value: n, fmt: (v) => `${v}`, aria: "seed 수", onInput: (v) => { n = v; update(); } });
  const grid = el("div", "param-grid");
  pn.row.style.gridTemplateColumns = "5.2em minmax(0, 1fr) 3.6em";
  grid.append(pn.row);
  const tools = el("div", "wframe-row");
  const reseed = el("button", "preset");
  reseed.type = "button";
  reseed.textContent = "↻ 다른 seed 묶음";
  reseed.addEventListener("click", () => { seed++; data = pool(seed); update(); });
  tools.append(reseed);
  const ro = readouts();
  f.panel.append(barBtns.row, modelBtns.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
