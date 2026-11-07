// 다단계 충전 Pareto (data/widgets.toml: charge-pareto).
// 30.6장의 가상 셀(3 Ah, 1-RC: R₀ 20 mΩ, R₁ 15 mΩ, C₁ 2000 F)에 집중 열 모델(45 J/K, 0.1 W/K, 주변 25 °C)을 붙인
// 2단계 CC 충전(SOC 10→50% 구간 c₁, 50→80% 구간 c₂)을 Δt = 2 s로 시뮬레이션한다. 장의 two_stage와 같은 갱신 순서다.
// 열화는 장의 가상 대리 지표 D = ∫ c² exp(3(SOC − 0.5) + (T − 25)/15) dt [h]다(계수는 지어낸 값).
// 1.0–4.0C(0.1C 간격) 961개 후보를 모두 평가해 전압 4.2 V·온도 45 °C 제약을 만족하는 점의 Pareto front를 그린다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

const Q_AH = 3.0, R0 = 0.02, R1 = 0.015, C1F = 2000.0;
const MCP = 45.0, HA = 0.1, T_AMB = 25.0, V_MAX = 4.2, T_MAX = 45.0;
const SOC0 = 0.1, SOC_SW = 0.5, SOC_END = 0.8, DT = 2.0;
const ocv = (s: number) => 3.5 + 0.55 * s + 0.04 * Math.log((s + 0.02) / (1.02 - s));
export const GRID = Array.from({ length: 31 }, (_, i) => 1.0 + i * 0.1);

export interface Result { c1: number; c2: number; tMin: number; wear: number; vPk: number; tPk: number; feasible: boolean; pareto: boolean }
export interface Trace { soc: number[]; v: number[]; temp: number[] }

/** 2단계 CC 한 후보. trace를 주면 step마다 SOC·단자 전압·온도를 기록한다. */
export function twoStage(c1: number, c2: number, trace?: Trace): Omit<Result, "feasible" | "pareto"> {
  const a = Math.exp(-DT / (R1 * C1F));
  let soc = SOC0, v1 = 0, temp = T_AMB, t = 0, wear = 0, vPk = 0, tPk = temp;
  while (soc < SOC_END) {
    const i = (soc < SOC_SW ? c1 : c2) * Q_AH;
    const v = ocv(soc) + i * R0 + v1;
    vPk = Math.max(vPk, v);
    if (trace) { trace.soc.push(soc); trace.v.push(v); trace.temp.push(temp); }
    wear += (i / Q_AH) ** 2 * Math.exp(3 * (soc - 0.5) + (temp - T_AMB) / 15) * DT / 3600;
    temp += (DT * (i * (i * R0 + v1) - HA * (temp - T_AMB))) / MCP;
    v1 = a * v1 + R1 * (1 - a) * i;
    t += DT; soc += (i * DT) / (3600 * Q_AH);
    tPk = Math.max(tPk, temp);
  }
  if (trace) { trace.soc.push(soc); trace.v.push(NaN); trace.temp.push(temp); }
  return { c1, c2, tMin: t / 60, wear, vPk, tPk };
}

/** 961개 후보 전체와 제약 만족 여부, Pareto 최적 여부 */
export function evaluateGrid(): Result[] {
  const all: Result[] = [];
  for (const c1 of GRID) for (const c2 of GRID) {
    const r = twoStage(c1, c2);
    all.push({ ...r, feasible: r.vPk <= V_MAX && r.tPk <= T_MAX, pareto: false });
  }
  const feas = all.filter((r) => r.feasible);
  for (const p of feas) {
    p.pareto = !feas.some((q) => q.tMin <= p.tMin && q.wear <= p.wear && (q.tMin < p.tMin || q.wear < p.wear));
  }
  return all;
}

const PRE = {
  single: { label: "1단계 1.9C", c1: 1.9, c2: 1.9 },
  best20: { label: "2.4→1.8C", c1: 2.4, c2: 1.8 },
  fast: { label: "3.1→1.9C", c1: 3.1, c2: 1.9 },
  reverse: { label: "역순 1.8→2.4C", c1: 1.8, c2: 2.4 },
} as const;
type PreKey = keyof typeof PRE;

export function mount(root: HTMLElement) {
  const wideLayout = (root.clientWidth || 600) >= 640;
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "다단계 충전 Pareto", "explore", wideLayout ? 0.5 : narrow ? 1.02 : 0.66,
    "2단계 충전 후보 961개의 충전 시간과 가상 열화 지표, Pareto front, 고른 프로토콜의 SOC별 전압·온도 곡선");
  const all = evaluateGrid();
  const front = all.filter((r) => r.pareto).sort((a, b) => a.tMin - b.tMin);
  const idx = (c: number) => Math.round((c - 1) * 10);
  const at = (c1: number, c2: number) => all[idx(c1) * 31 + idx(c2)];
  let c1 = 2.4, c2 = 1.8;
  let sel = at(c1, c2);
  let tr: Trace = { soc: [], v: [], temp: [] };

  const tmin = 8, tmax = 44; // 분, 축 고정(후보 961개가 모두 들어간다: 10.5–42.0 min)
  const wlo = Math.log(0.6), whi = Math.log(Math.max(...all.map((r) => r.wear)) * 1.1);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pad = 8;
    let A: { x: number; y: number; w: number; h: number }, Vb: typeof A, Tb: typeof A;
    if (wideLayout) {
      const lw = Math.round(W * 0.56);
      A = { x: 50, y: 26, w: lw - 50 - 14, h: H - 26 - 34 - 20 };
      const rx = lw + 40, rw = W - rx - 14, hh = (H - 26 - 34 - 34) / 2;
      Vb = { x: rx, y: 26, w: rw, h: hh };
      Tb = { x: rx, y: 26 + hh + 34, w: rw, h: hh };
    } else {
      const ah = Math.round((H - 26 - 30) * 0.6);
      A = { x: 46, y: 24, w: W - 46 - 12, h: ah - 10 - 18 };
      const by = 24 + ah + 34, bh = H - by - 24, half = (W - 46 - 12 - 40) / 2;
      Vb = { x: 46, y: by, w: half, h: bh };
      Tb = { x: 46 + half + 40, y: by, w: half, h: bh };
    }
    void pad;
    const mono = "11px JetBrains Mono, monospace", sans = "600 12px Pretendard Variable, sans-serif";
    // ── Pareto 그림
    const X = (t: number) => A.x + ((t - tmin) / (tmax - tmin)) * A.w;
    const Y = (w: number) => A.y + A.h - ((Math.log(w) - wlo) / (whi - wlo)) * A.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(A.x + 0.5, A.y + 0.5, A.w - 1, A.h - 1);
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const w of [1, 2, 5, 10, 20]) {
      if (Math.log(w) < wlo || Math.log(w) > whi) continue;
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(A.x, Y(w)); ctx.lineTo(A.x + A.w, Y(w)); ctx.stroke();
      ctx.fillText(String(w), A.x - 5, Y(w) + 4);
    }
    ctx.textAlign = "center";
    for (let t = 10; t <= 40; t += 5) ctx.fillText(String(t), X(t), A.y + A.h + 14);
    ctx.textAlign = "left"; ctx.font = sans;
    ctx.fillText("충전 시간 10→80% (min) · 열화 지표(가상, 로그 축)", A.x, A.y - 9);
    ctx.save(); ctx.beginPath(); ctx.rect(A.x, A.y, A.w, A.h); ctx.clip();
    for (const r of all) {
      if (r.feasible) continue;
      ctx.strokeStyle = alpha(C.mid, 0.45); ctx.lineWidth = 1;
      const x = X(r.tMin), y = Y(r.wear);
      ctx.beginPath(); ctx.moveTo(x - 2, y - 2); ctx.lineTo(x + 2, y + 2); ctx.moveTo(x - 2, y + 2); ctx.lineTo(x + 2, y - 2); ctx.stroke();
    }
    for (const r of all) {
      if (!r.feasible) continue;
      ctx.fillStyle = alpha(C.ink2, 0.75);
      ctx.beginPath(); ctx.arc(X(r.tMin), Y(r.wear), 2.4, 0, 7); ctx.fill();
    }
    // 1단계(같은 C-rate) 후보: 고리
    for (const r of all) {
      if (Math.abs(r.c1 - r.c2) > 1e-9 || !r.feasible) continue;
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.arc(X(r.tMin), Y(r.wear), 4.5, 0, 7); ctx.stroke();
    }
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath();
    front.forEach((r, i) => (i ? ctx.lineTo(X(r.tMin), Y(r.wear)) : ctx.moveTo(X(r.tMin), Y(r.wear))));
    ctx.stroke();
    // 고른 점
    const sx = X(sel.tMin), sy = Y(sel.wear);
    ctx.fillStyle = sel.feasible ? C.edge : C.bg; ctx.strokeStyle = C.edge; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, 7); ctx.fill(); ctx.stroke();
    ctx.restore();
    // 범례
    // 범례: x축 눈금 아래 한 줄
    ctx.font = mono; ctx.textAlign = "left";
    const ly = A.y + A.h + 31;
    let lx = A.x;
    const legend = (label: string, mark: (x: number, y: number) => void) => {
      mark(lx + 4, ly - 4); ctx.fillStyle = C.mid; ctx.fillText(label, lx + 13, ly); lx += ctx.measureText(label).width + 26;
    };
    legend("제약 만족", (x, y) => { ctx.fillStyle = alpha(C.ink2, 0.75); ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.fill(); });
    legend("위반", (x, y) => { ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x - 3, y - 3); ctx.lineTo(x + 3, y + 3); ctx.moveTo(x - 3, y + 3); ctx.lineTo(x + 3, y - 3); ctx.stroke(); });
    legend("1단계", (x, y) => { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.stroke(); });
    legend(A.w < 330 ? "front" : "Pareto front", (x, y) => { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 6, y); ctx.lineTo(x + 3, y); ctx.stroke(); });

    // ── SOC별 전압·온도
    const n = tr.soc.length;
    const XS = (b: typeof A, s: number) => b.x + ((s - SOC0) / (SOC_END - SOC0)) * b.w;
    const panel = (b: typeof A, vals: number[], lo: number, hi: number, lim: number, title: string, ticks: number[], fmt: (v: number) => string) => {
      const Yb = (v: number) => b.y + b.h - ((v - lo) / (hi - lo)) * b.h;
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
      for (const v of ticks) ctx.fillText(fmt(v), b.x - 4, Yb(v) + 4);
      ctx.textAlign = "center";
      for (const s of [0.1, 0.5, 0.8]) ctx.fillText(`${Math.round(s * 100)}%`, Math.min(Math.max(XS(b, s), b.x + 12), b.x + b.w - 12), b.y + b.h + 14);
      // 단계 경계
      ctx.setLineDash([2, 3]); ctx.strokeStyle = C.mid;
      ctx.beginPath(); ctx.moveTo(XS(b, SOC_SW), b.y); ctx.lineTo(XS(b, SOC_SW), b.y + b.h); ctx.stroke();
      // 한계선
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.edge; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(b.x, Yb(lim)); ctx.lineTo(b.x + b.w, Yb(lim)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      // 곡선: 한계 이하 = 보조색, 초과 = 강조색
      ctx.lineWidth = 2.2;
      for (let k = 1; k < n; k++) {
        if (!Number.isFinite(vals[k]) || !Number.isFinite(vals[k - 1])) continue;
        ctx.strokeStyle = vals[k] > lim ? C.edge : C.ink2;
        ctx.beginPath(); ctx.moveTo(XS(b, tr.soc[k - 1]), Yb(vals[k - 1])); ctx.lineTo(XS(b, tr.soc[k]), Yb(vals[k])); ctx.stroke();
      }
      // 첫 위반 위치
      const kv = vals.findIndex((v) => v > lim);
      if (kv >= 0) {
        const x = XS(b, tr.soc[kv]);
        ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(x, Yb(lim), 4.5, 0, 7); ctx.fill();
      }
      ctx.restore();
      ctx.textAlign = "left"; ctx.font = sans; ctx.fillStyle = C.mid;
      ctx.fillText(title, b.x, b.y - 8);
    };
    panel(Vb, tr.v, 3.6, 4.4, V_MAX, "단자 전압 (V)", [3.6, 4.0, 4.2, 4.4], (v) => v.toFixed(1));
    panel(Tb, tr.temp, 25, 60, T_MAX, "온도 (°C)", [25, 45, 60], (v) => String(v));
  }

  function update() {
    sel = at(c1, c2);
    tr = { soc: [], v: [], temp: [] };
    twoStage(c1, c2, tr);
    const fmtSoc = (k: number) => `${Math.round(tr.soc[k] * 100)}%`;
    const kv = tr.v.findIndex((v) => v > V_MAX), kt = tr.temp.findIndex((v) => v > T_MAX);
    ro.set("t", `${sel.tMin.toFixed(1)} min · 열화(가상) ${sel.wear.toFixed(3)}`);
    ro.set("pk", `최고 ${sel.vPk.toFixed(3)} V · ${sel.tPk.toFixed(1)} °C`);
    if (!sel.feasible) {
      const why = [kv >= 0 ? `4.2 V 초과(SOC ${fmtSoc(kv)})` : "", kt >= 0 ? `45 °C 초과(SOC ${fmtSoc(kt)})` : ""].filter(Boolean).join(" · ");
      ro.set("st", `제약 위반: ${why}`, "hot");
    } else if (sel.pareto) {
      ro.set("st", "Pareto 최적", "ok");
    } else {
      // 이 점을 지배하는 front 점 중 시간이 가장 가까운 것
      const dom = front.filter((q) => q.tMin <= sel.tMin && q.wear <= sel.wear);
      const q = dom.reduce((b, r) => (Math.abs(r.tMin - sel.tMin) < Math.abs(b.tMin - sel.tMin) ? r : b), dom[0]);
      ro.set("st", `지배당함: ${q.c1.toFixed(1)}C → ${q.c2.toFixed(1)}C가 ${q.tMin.toFixed(1)} min · ${q.wear.toFixed(3)}`, "hot");
    }
    pageState.set("chargePareto", { c1, c2, t_min: sel.tMin, wear: sel.wear, v_peak_V: sel.vPk, temp_peak_C: sel.tPk, feasible: sel.feasible, pareto: sel.pareto }, root);
    draw();
  }

  const fmt = (v: number) => `${v.toFixed(1)}C`;
  const p1 = param({ label: "1단계 c₁", min: 1, max: 4, step: 0.1, value: c1, fmt, aria: "SOC 10–50% 구간의 C-rate", onInput: (v) => { c1 = v; pre.select(null); update(); } });
  const p2 = param({ label: "2단계 c₂", min: 1, max: 4, step: 0.1, value: c2, fmt, aria: "SOC 50–80% 구간의 C-rate", onInput: (v) => { c2 = v; pre.select(null); update(); } });
  for (const p of [p1, p2]) p.row.style.gridTemplateColumns = "5.4em minmax(0, 1fr) 3em";
  const grid = el("div", "param-grid");
  grid.append(p1.row, p2.row);
  const pre = presets<PreKey>((Object.keys(PRE) as PreKey[]).map((k) => ({ key: k, label: PRE[k].label })), (k) => {
    c1 = PRE[k].c1; c2 = PRE[k].c2; p1.set(c1); p2.set(c2); update();
  }, "best20");
  pre.row.classList.add("segmented");
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 9px"));
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
