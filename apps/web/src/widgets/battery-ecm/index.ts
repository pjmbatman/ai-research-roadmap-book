// 등가회로 모델 시뮬레이터 (data/widgets.toml: battery-ecm).
// 1-RC 등가회로 모델(ECM). 전류 I > 0은 방전이다.
//   V(t) = OCV(SOC) − I·R₀ − V₁,   dV₁/dt = −V₁/(R₁C₁) + I/C₁,   dSOC/dt = −I/(3600·Q)
// 전류가 한 step(Δt = 1 s) 동안 일정하다고 보고 V₁을 정확히 적분한다(영차 유지):
//   V₁ ← V₁·e^{−Δt/τ} + R₁(1 − e^{−Δt/τ})·I,   τ = R₁C₁
// 전류가 바뀌는 순간 R₀ 때문에 전압이 ΔI·R₀만큼 즉시 뛰고, 그 뒤 RC 쌍이 시상수 τ로 천천히 따라온다.
// OCV 곡선과 용량 Q = 3 Ah는 설명용 값이다(특정 셀의 측정값이 아니다).
import { frame, param, presets, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

const Q_AH = 3.0, SOC0 = 0.8, DT = 1, T_END = 900;
const ocv = (s: number) => 3.5 + 0.55 * s + 0.04 * Math.log((s + 0.02) / (1.02 - s));
type Profile = "pulse" | "steps" | "dynamic";
const PROFILES: Record<Profile, { label: string; I: (t: number) => number }> = {
  pulse: { label: "방전 펄스", I: (t) => (t >= 60 && t < 360 ? 3 : 0) },
  steps: { label: "계단 부하", I: (t) => (t < 60 ? 0 : t < 240 ? 1.5 : t < 420 ? 3 : t < 600 ? 6 : 0) },
  dynamic: {
    label: "가변 부하(회생 포함)",
    I: (() => {
      const r = rng(2024), seg = Array.from({ length: Math.ceil(T_END / 20) }, () => Math.round((-3 + 9 * r.u()) * 2) / 2);
      return (t: number) => (t < 40 || t >= 780 ? 0 : seg[Math.floor(t / 20)]);
    })(),
  },
};

interface Sim { I: Float64Array; V: Float64Array; O: Float64Array; Vr0: Float64Array; soc: Float64Array }
function simulate(p: Profile, R0: number, R1: number, C1: number): Sim {
  const n = T_END / DT + 1;
  const s: Sim = { I: new Float64Array(n), V: new Float64Array(n), O: new Float64Array(n), Vr0: new Float64Array(n), soc: new Float64Array(n) };
  const tau = R1 * C1, a = tau > 0 ? Math.exp(-DT / tau) : 0;
  let v1 = 0, soc = SOC0;
  for (let k = 0; k < n; k++) {
    const I = PROFILES[p].I(k * DT);
    const o = ocv(soc);
    s.I[k] = I; s.O[k] = o; s.Vr0[k] = o - I * R0; s.V[k] = o - I * R0 - v1; s.soc[k] = soc;
    v1 = v1 * a + R1 * (1 - a) * I;
    soc -= (I * DT) / (3600 * Q_AH);
  }
  return s;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "등가회로 모델 시뮬레이터", "simulate", narrow ? 0.95 : 0.6, "전류 프로파일과 1-RC 등가회로 모델의 단자 전압, 개방 전압 곡선");
  let profile: Profile = "pulse", R0 = 20, R1 = 15, C1 = 2.0; // mΩ, mΩ, kF
  let sim = simulate(profile, R0 / 1000, R1 / 1000, C1 * 1000);
  let range = [0, 1];

  /** 프로파일마다 고정한 전압 축: 저항이 가장 클 때도 그림 안에 들어오게 한다(슬라이더를 움직여도 축이 흔들리지 않는다). */
  function voltRange(p: Profile) {
    let lo = Infinity, hi = -Infinity;
    for (const [r0, r1] of [[0, 0], [0.05, 0.05]]) {
      const s = simulate(p, r0, r1, 200);
      for (let k = 0; k < s.V.length; k++) { lo = Math.min(lo, s.V[k], s.O[k]); hi = Math.max(hi, s.V[k], s.O[k]); }
    }
    const m = (hi - lo) * 0.08;
    return [lo - m, hi + m];
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const padL = 44, padR = 14, top = 24, bot = 24, gap = 30;
    const iH = Math.round((H - top - bot - gap) * 0.3), vH = H - top - bot - gap - iH;
    const Ib = { x: padL, y: top, w: W - padL - padR, h: iH };
    const Vb = { x: padL, y: top + iH + gap, w: W - padL - padR, h: vH };
    const n = sim.V.length;
    const X = (k: number) => Ib.x + (k / (n - 1)) * Ib.w;
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.textAlign = "left";

    // 전류
    const imax = Math.max(3, ...Array.from(sim.I, Math.abs)) * 1.1, imin = Math.min(0, ...Array.from(sim.I)) * 1.1;
    const lo = Math.min(imin, -0.1 * imax);
    const YI = (v: number) => Ib.y + Ib.h - ((v - lo) / (imax - lo)) * Ib.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(Ib.x + 0.5, Ib.y + 0.5, Ib.w - 1, Ib.h - 1);
    ctx.beginPath(); ctx.moveTo(Ib.x, YI(0)); ctx.lineTo(Ib.x + Ib.w, YI(0)); ctx.stroke();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k < n; k++) { const y = YI(sim.I[k]); if (k) { ctx.lineTo(X(k), YI(sim.I[k - 1])); ctx.lineTo(X(k), y); } else ctx.moveTo(X(k), y); }
    ctx.stroke();
    ctx.fillStyle = C.mid;
    ctx.fillText("전류 I (A, 방전 +)", Ib.x, Ib.y - 8);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "right";
    for (const v of [0, Math.round(imax / 1.1)]) ctx.fillText(String(v), Ib.x - 6, YI(v) + 4);

    // 전압
    const [vlo, vhi] = range;
    const YV = (v: number) => Vb.y + Vb.h - ((v - vlo) / (vhi - vlo)) * Vb.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(Vb.x + 0.5, Vb.y + 0.5, Vb.w - 1, Vb.h - 1);
    const step = vhi - vlo > 0.6 ? 0.2 : 0.1;
    ctx.fillStyle = C.mid;
    for (let v = Math.ceil(vlo / step) * step; v <= vhi; v += step) {
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(Vb.x, YV(v)); ctx.lineTo(Vb.x + Vb.w, YV(v)); ctx.stroke();
      ctx.fillText(v.toFixed(1), Vb.x - 6, YV(v) + 4);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(Vb.x, Vb.y, Vb.w, Vb.h); ctx.clip();
    // OCV와 V 사이(과전압): R₀ 몫과 RC 몫
    const band = (a: Float64Array, b: Float64Array, fill: string) => {
      ctx.fillStyle = fill; ctx.beginPath();
      for (let k = 0; k < n; k++) (k ? ctx.lineTo(X(k), YV(a[k])) : ctx.moveTo(X(k), YV(a[k])));
      for (let k = n - 1; k >= 0; k--) ctx.lineTo(X(k), YV(b[k]));
      ctx.closePath(); ctx.fill();
    };
    band(sim.O, sim.Vr0, C.grid2);
    band(sim.Vr0, sim.V, C.fill);
    const curve = (a: Float64Array, col: string, lw: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash);
      ctx.beginPath(); for (let k = 0; k < n; k++) (k ? ctx.lineTo(X(k), YV(a[k])) : ctx.moveTo(X(k), YV(a[k]))); ctx.stroke();
      ctx.setLineDash([]);
    };
    curve(sim.O, C.mid, 1.4, [5, 4]);
    curve(sim.V, C.edge, 2.4);
    ctx.restore();
    ctx.textAlign = "left"; ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("단자 전압 V와 OCV (V)", Vb.x, Vb.y - 8);
    // 시간 축
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "center";
    for (const s of [0, 300, 600, 900]) ctx.fillText(s === 900 ? "900 s" : String(s), Math.min(X(s / DT), W - padR - 18), Vb.y + Vb.h + 16);
    ctx.textAlign = "left";
  }

  function update() {
    sim = simulate(profile, R0 / 1000, R1 / 1000, C1 * 1000);
    const tau = (R1 / 1000) * (C1 * 1000);
    // 첫 전류 변화 순간의 전압 변화
    let jump = 0, k0 = -1;
    for (let k = 1; k < sim.I.length; k++) if (sim.I[k] !== sim.I[k - 1]) { k0 = k; break; }
    if (k0 > 0) jump = sim.V[k0] - sim.V[k0 - 1];
    const vmin = Math.min(...Array.from(sim.V));
    ro.set("tau", `τ = R₁C₁ = ${tau < 10 ? tau.toFixed(1) : tau.toFixed(0)} s`, "hot");
    ro.set("jump", `첫 변화의 순간 ΔV ${(jump * 1000).toFixed(0)} mV`);
    ro.set("vmin", `최저 전압 ${vmin.toFixed(3)} V`);
    ro.set("soc", `SOC ${(SOC0 * 100).toFixed(0)}% → ${(sim.soc[sim.soc.length - 1] * 100).toFixed(1)}%`);
    pageState.set("ecm", { profile, R0_ohm: R0 / 1000, R1_ohm: R1 / 1000, C1_F: C1 * 1000, tau_s: tau, vmin_V: vmin }, root);
    draw();
  }

  const profBtns = presets<Profile>((Object.keys(PROFILES) as Profile[]).map((k) => ({ key: k, label: PROFILES[k].label })), (k) => {
    profile = k; range = voltRange(k); update();
  }, profile);
  profBtns.row.classList.add("segmented");
  if (narrow) profBtns.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 9px"));
  const pR0 = param({ label: "R₀", min: 0, max: 50, step: 1, value: R0, fmt: (v) => `${v} mΩ`, aria: "직렬 저항 R₀ (밀리옴)", onInput: (v) => { R0 = v; update(); } });
  const pR1 = param({ label: "R₁", min: 0, max: 50, step: 1, value: R1, fmt: (v) => `${v} mΩ`, aria: "RC 쌍의 저항 R₁ (밀리옴)", onInput: (v) => { R1 = v; update(); } });
  const pC1 = param({ label: "C₁", min: 0.2, max: 10, step: 0.1, value: C1, fmt: (v) => `${v < 9.95 ? v.toFixed(1) : v.toFixed(0)} kF`, aria: "RC 쌍의 커패시턴스 C₁ (킬로패럿)", onInput: (v) => { C1 = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pR0.row, pR1.row, pC1.row);
  const ro = readouts();
  f.panel.append(profBtns.row, grid, ro.row);
  range = voltRange(profile);
  f.onResize(draw);
  update();
}
