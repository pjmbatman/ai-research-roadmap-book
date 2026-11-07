// Kaplan–Meier와 중도절단 (data/widgets.toml: kaplan-meier).
// 16.8장 셀과 같은 셀 300개: 참 수명 T_i ~ Weibull(3, 1000 cycle), 시험 종료 시점 C_i⁰ ~ U(400, 1200).
// 중도절단 비율 슬라이더는 모든 셀의 시험 종료 시점을 같은 배율 κ로 늘이거나 줄여(C_i = κ·C_i⁰) 그 비율을 맞춘다.
// 배율은 중도절단과 무관한 상수이므로 독립 중도절단 가정이 그대로 성립한다. 비율 0.55에서 κ = 1(본문과 같은 데이터)이다.
// 추정량: Kaplan–Meier Ŝ(t) = Π_{t_j ≤ t}(1 − d_j/n_j), 중도절단 버림 = 고장 관측만의 생존 비율, 중도절단=고장 = 관측 시간 전체의 생존 비율.
// 참값 S(t) = exp(−(t/1000)³), 참 중앙 수명 = 1000·(ln 2)^{1/3} ≈ 885 cycle.
import { frame, param, presets, readouts, C, alpha, el } from "../frame";
import { pageState } from "../../client/store";
import { LIFE, STOP } from "./data";

const n = LIFE.length;
const RATIO = LIFE.map((l, i) => l / STOP[i]).sort((a, b) => a - b);
export const trueS = (t: number) => Math.exp(-((t / 1000) ** 3));
export const TRUE_MED = 1000 * Math.log(2) ** (1 / 3);
type Est = "km" | "drop" | "fail";

/** 중도절단 비율 p에 맞는 배율 κ와 관측 (time, event) */
export function observe(p: number) {
  const m = n - Math.round(p * n); // 고장 관측 수
  const lo = m > 0 ? RATIO[m - 1] : 0, hi = m < n ? RATIO[m] : Infinity;
  const kappa = Math.min(Math.max(1, lo), lo + 0.999 * (hi - lo)); // 고장 관측 m개를 만드는 κ ∈ [lo, hi) 중 1에 가장 가까운 값
  const time = LIFE.map((l, i) => Math.min(l, kappa * STOP[i]));
  const event = LIFE.map((l, i) => l <= kappa * STOP[i]);
  return { kappa, time, event, nev: m, tmax: Math.max(...time) };
}
type Obs = ReturnType<typeof observe>;

/** Kaplan–Meier: 고장 시각 t_j와 Ŝ(t_j) */
export function kaplanMeier(o: Obs) {
  const tj = [...new Set(o.time.filter((_, i) => o.event[i]))].sort((a, b) => a - b);
  let s = 1;
  const S = tj.map((t) => {
    let d = 0, r = 0;
    for (let i = 0; i < n; i++) { if (o.time[i] >= t) r++; if (o.time[i] === t && o.event[i]) d++; }
    s *= 1 - d / r;
    return s;
  });
  return { tj, S };
}
/** Python의 "{:.0f}"처럼 반올림(절반은 짝수 쪽) */
const round0 = (v: number) => { const f = Math.floor(v), d = v - f; return String(d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f); };
const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q), k = s.length; return k % 2 ? s[(k - 1) / 2] : 0.5 * (s[k / 2 - 1] + s[k / 2]); };

/** 세 추정량의 생존함수(계단)와 중앙 수명 */
export function estimates(o: Obs) {
  const km = kaplanMeier(o);
  const evT = o.time.filter((_, i) => o.event[i]);
  const kmAt = (t: number) => { let v = 1; for (let j = 0; j < km.tj.length && km.tj[j] <= t; j++) v = km.S[j]; return v; };
  const dropAt = (t: number) => evT.filter((v) => v > t).length / evT.length;
  const failAt = (t: number) => o.time.filter((v) => v > t).length / n;
  const kmMedIdx = km.S.findIndex((v) => v <= 0.5);
  return {
    km, at: { km: kmAt, drop: dropAt, fail: failAt } as Record<Est, (t: number) => number>,
    med: { km: kmMedIdx >= 0 ? km.tj[kmMedIdx] : NaN, drop: median(evT), fail: median(o.time) } as Record<Est, number>,
  };
}

export function mount(root: HTMLElement) {
  const f = frame(root, "Kaplan–Meier와 중도절단", "manipulate", 0.56, "셀 300개의 생존함수: 참 생존함수, Kaplan–Meier 추정, 중도절단을 버리거나 고장으로 친 단순 비율");
  let p = 0.55, est: Est = "km";
  let obs = observe(p), E = estimates(obs);
  const NAME: Record<Est, string> = { km: "Kaplan–Meier", drop: "중도절단 버림", fail: "중도절단=고장" };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const PL = 34, PR = 12, top = 24, bottom = H - 24, T1 = 1800;
    const X = (t: number) => PL + (Math.min(t, T1) / T1) * (W - PL - PR);
    const Y = (s: number) => bottom - s * (bottom - top);
    // 관측 기간 밖
    if (obs.tmax < T1) {
      ctx.fillStyle = alpha(C.mid, 0.12); ctx.fillRect(X(obs.tmax), top, W - PR - X(obs.tmax), bottom - top);
      ctx.fillStyle = C.mid; const s = "관측 밖"; ctx.fillText(s, Math.min(X(obs.tmax) + 6, W - PR - ctx.measureText(s).width - 2), top + 12);
    }
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(PL, Y(v)); ctx.lineTo(W - PR, Y(v)); ctx.stroke(); ctx.fillStyle = C.mid; ctx.fillText(v.toFixed(1), PL - 6 - ctx.measureText(v.toFixed(1)).width, Y(v) + 4); }
    for (const t of [0, 400, 800, 1200]) ctx.fillText(String(t), X(t) - ctx.measureText(String(t)).width / 2, bottom + 15);
    ctx.fillText("S(t)", PL, top - 9);
    const u = "cycle"; ctx.fillText(u, W - PR - ctx.measureText(u).width, bottom + 15);
    // 참 생존함수
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); for (let t = 0; t <= T1; t += 10) (t ? ctx.lineTo(X(t), Y(trueS(t))) : ctx.moveTo(X(t), Y(1))); ctx.stroke(); ctx.setLineDash([]);
    // 계단 곡선
    const step = (at: (t: number) => number, col: string, w: number, upto: number) => {
      ctx.strokeStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(X(0), Y(1));
      let prev = 1;
      for (let t = 1; t <= Math.min(upto, T1); t += 2) { const v = at(t); if (v !== prev) { ctx.lineTo(X(t), Y(prev)); ctx.lineTo(X(t), Y(v)); prev = v; } }
      ctx.lineTo(X(Math.min(upto, T1)), Y(prev)); ctx.stroke();
    };
    if (est !== "km") step(E.at.km, alpha(C.ink2, 0.7), 1.4, obs.tmax);
    step(E.at[est], C.edge, 2.4, est === "drop" ? Math.max(...obs.time.filter((_, i) => obs.event[i])) : obs.tmax);
    // 중도절단 표시(KM 곡선 위의 짧은 세로 눈금)
    if (est === "km") {
      ctx.strokeStyle = alpha(C.edge, 0.6); ctx.lineWidth = 1;
      obs.time.forEach((t, i) => { if (obs.event[i] || t > T1) return; const y = Y(E.at.km(t)); ctx.beginPath(); ctx.moveTo(X(t), y - 4); ctx.lineTo(X(t), y + 4); ctx.stroke(); });
    }
    // 중앙 수명 위치
    const med = E.med[est];
    if (Number.isFinite(med)) { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X(med), Y(0.5), 4.5, 0, 7); ctx.fill(); }
    ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(TRUE_MED), Y(0.5), 3.5, 0, 7); ctx.fill();
    // 범례
    let lx = PL + 34;
    const items: [string, string, boolean][] = [["참 S(t)", C.ink, true], [NAME[est], C.edge, false]];
    if (est !== "km") items.push(["KM", alpha(C.ink2, 0.7), false]);
    for (const [nm, col, dash] of items) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, top - 13); ctx.lineTo(lx + 14, top - 13); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, top - 9);
      lx += 30 + ctx.measureText(nm).width;
    }
  }

  let raf = 0;
  function update() {
    obs = observe(p);
    E = estimates(obs);
    const med = E.med[est], bias = (med - TRUE_MED) / TRUE_MED;
    ro.set("n", `고장 ${obs.nev} · 중도절단 ${n - obs.nev}개 (${Math.round((100 * (n - obs.nev)) / n)}%)`);
    ro.set("med", Number.isFinite(med) ? `중앙 수명 ${round0(med)} cycle (참 ${round0(TRUE_MED)})` : "중앙 수명: 관측 기간 안에서 S가 0.5까지 내려가지 않는다", Number.isFinite(med) ? (Math.abs(bias) < 0.05 ? "ok" : "hot") : "");
    ro.set("bias", Number.isFinite(med) ? `편향 ${bias >= 0 ? "+" : "−"}${Math.abs(100 * bias).toFixed(0)}%` : "편향 —");
    ro.set("s800", `S(800): 참 ${trueS(800).toFixed(3)} · 추정 ${E.at[est](800).toFixed(3)}`);
    pageState.set("kaplan-meier", { censored: p, estimator: est, median: med }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pe = presets<Est>([{ key: "km", label: "KM" }, { key: "drop", label: "절단 버림" }, { key: "fail", label: "절단=고장" }], (k) => { est = k; update(); }, est);
  pe.row.classList.add("segmented");
  const pp = param({ label: "절단", min: 0, max: 0.9, step: 0.01, value: p, fmt: (v) => `${Math.round(v * 100)}%`, aria: "중도절단 비율 (시험 종료 시점을 모든 셀에서 같은 배율로 줄이거나 늘인다)", onInput: (v) => { p = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pp.row);
  const ro = readouts();
  f.panel.append(pe.row, grid, ro.row);
  f.onResize(draw);
  update();
}
