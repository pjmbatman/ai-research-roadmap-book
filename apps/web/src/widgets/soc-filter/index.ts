// 쿨롱 카운팅 vs EKF (data/widgets.toml: soc-filter).
// 30.4장의 설정을 그대로 쓴다: 실제 용량 2.7 Ah로 열화된 셀, 20 s마다 바뀌는 가변 부하 U(−2, 3.6) A로 2시간(Δt = 1 s),
// 참 SOC 90%에서 시작. 1-RC 모델(R₀ 20 mΩ, R₁ 15 mΩ, C₁ 2000 F, 가상의 예시 셀)로 측정 전압을 만들고 5 mV 잡음을 더한다.
// 추정기는 센서 offset이 섞인 전류와 (스위치를 켜면) 공칭 용량 3.0 Ah를 쓴다.
//   쿨롱 카운팅: SOC_k = SOC₀ − Σ I_meas Δt / (3600 Q)
//   EKF: 상태 (SOC, V₁), 관측 V = OCV(SOC) − I R₀ − V₁, H = [dOCV/dSOC, −1] — 장의 코드와 같은 갱신 순서·잡음 공분산.
// OCV 기울기 배율 κ: OCV_κ(s) = 3.775 + κ·(0.55(s − 0.5) + 0.04 ln((s + 0.02)/(1.02 − s))). κ = 1이면 장의 OCV 곡선이다.
import { frame, param, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

const DT = 1, T = 7200, Q_TRUE = 2.7, Q_NOM = 3.0, SOC_TRUE0 = 0.9;
const R0 = 0.02, R1 = 0.015, C1F = 2000, A1 = Math.exp(-DT / (R1 * C1F)), NOISE_V = 0.005;

/** 고정 시드의 부하 전류(A, 방전 +)와 표준정규 잡음 */
export function inputs(seed = 30) {
  const r = rng(seed);
  const iTrue = new Float64Array(T), z = new Float64Array(T);
  for (let s = 0; s < T / 20; s++) { const v = -2 + 5.6 * r.u(); for (let j = 0; j < 20; j++) iTrue[s * 20 + j] = v; }
  for (let k = 0; k < T; k++) z[k] = r.normal();
  return { iTrue, z };
}

export interface Opts { offsetA: number; soc0Est: number; kappa: number; nominalQ: boolean }
export function simulate(inp: ReturnType<typeof inputs>, o: Opts) {
  const { iTrue, z } = inp;
  const ocv = (s: number) => 3.775 + o.kappa * (0.55 * (s - 0.5) + 0.04 * Math.log((s + 0.02) / (1.02 - s)));
  const docv = (s: number) => o.kappa * (0.55 + 0.04 * (1 / (s + 0.02) + 1 / (1.02 - s)));
  const qModel = o.nominalQ ? Q_NOM : Q_TRUE;
  const socTrue = new Float64Array(T), vMeas = new Float64Array(T);
  let acc = 0;
  for (let k = 0; k < T; k++) { acc += iTrue[k]; socTrue[k] = SOC_TRUE0 - (acc * DT) / 3600 / Q_TRUE; }
  let v1 = 0;
  for (let k = 0; k < T; k++) {
    vMeas[k] = ocv(socTrue[k]) - iTrue[k] * R0 - v1 + NOISE_V * z[k];
    v1 = A1 * v1 + R1 * (1 - A1) * iTrue[k];
  }
  // 쿨롱 카운팅
  const cc = new Float64Array(T);
  acc = 0;
  for (let k = 0; k < T; k++) { acc += iTrue[k] + o.offsetA; cc[k] = o.soc0Est - (acc * DT) / 3600 / qModel; }
  // EKF
  const ekf = new Float64Array(T), sd = new Float64Array(T), gain = new Float64Array(T);
  let x0 = o.soc0Est, x1 = 0;
  let p00 = 0.1 ** 2, p01 = 0, p11 = 0.01 ** 2;
  const q00 = 1e-7, q11 = 1e-8, rn = NOISE_V ** 2;
  for (let k = 0; k < T; k++) {
    const i = iTrue[k] + o.offsetA;
    const h0 = docv(x0), h1 = -1;
    const ph0 = p00 * h0 + p01 * h1, ph1 = p01 * h0 + p11 * h1; // P Hᵀ
    const S = h0 * ph0 + h1 * ph1 + rn;
    const k0 = ph0 / S, k1 = ph1 / S;
    const innov = vMeas[k] - (ocv(x0) - i * R0 - x1);
    x0 += k0 * innov; x1 += k1 * innov;
    // P ← P − K (H P)
    const hp0 = h0 * p00 + h1 * p01, hp1 = h0 * p01 + h1 * p11;
    p00 -= k0 * hp0; p01 -= k0 * hp1; p11 -= k1 * hp1;
    ekf[k] = x0; sd[k] = Math.sqrt(Math.max(p00, 0)); gain[k] = k0;
    // 예측
    x0 = x0 - (i * DT) / 3600 / qModel;
    x1 = A1 * x1 + R1 * (1 - A1) * i;
    p00 = p00 + q00; p01 = A1 * p01; p11 = A1 * A1 * p11 + q11;
  }
  return { socTrue, cc, ekf, sd, gain, slopeMid: docv(0.5) };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "쿨롱 카운팅 vs EKF", "simulate", narrow ? 0.85 : 0.52, "2시간 동안의 참 SOC, 쿨롱 카운팅 추정, EKF 추정과 두 추정의 오차 곡선");
  const inp = inputs();
  let offset = 0.05, err0 = -30, kappa = 1, nominal = true;
  let sim = simulate(inp, { offsetA: offset, soc0Est: SOC_TRUE0 + err0 / 100, kappa, nominalQ: nominal });

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const L = 42, R = 12, top = 24, bot = 22, gap = 30;
    const h1 = Math.round((H - top - bot - gap) * 0.5), h2 = H - top - bot - gap - h1;
    const A = { x: L, y: top, w: W - L - R, h: h1 }, B = { x: L, y: top + h1 + gap, w: W - L - R, h: h2 };
    const X = (k: number) => A.x + (k / (T - 1)) * A.w;
    const mono = "11px JetBrains Mono, monospace", sans = "600 12px Pretendard Variable, sans-serif";
    const step = Math.max(1, Math.floor(T / (A.w * 1.5)));
    const curve = (arr: Float64Array, Y: (v: number) => number, col: string, lw: number, dash: number[] = [], scale = 100) => {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash);
      ctx.beginPath();
      for (let k = 0; k < T; k += step) (k ? ctx.lineTo(X(k), Y(arr[k] * scale)) : ctx.moveTo(X(k), Y(arr[k] * scale)));
      ctx.lineTo(X(T - 1), Y(arr[T - 1] * scale));
      ctx.stroke(); ctx.setLineDash([]);
    };
    // ── SOC
    const sl = 0, sh = 100;
    const YA = (v: number) => A.y + A.h - ((v - sl) / (sh - sl)) * A.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(A.x + 0.5, A.y + 0.5, A.w - 1, A.h - 1);
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 50, 100]) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(A.x, YA(v)); ctx.lineTo(A.x + A.w, YA(v)); ctx.stroke(); ctx.fillText(String(v), A.x - 5, YA(v) + 4); }
    ctx.save(); ctx.beginPath(); ctx.rect(A.x, A.y, A.w, A.h); ctx.clip();
    curve(sim.socTrue, YA, C.ink, 2.6);
    curve(sim.cc, YA, C.ink2, 2, [6, 4]);
    curve(sim.ekf, YA, C.edge, 1.8);
    ctx.restore();
    ctx.textAlign = "left"; ctx.font = sans; ctx.fillStyle = C.mid;
    ctx.fillText("SOC (%)", A.x, A.y - 8);
    // 범례
    ctx.font = mono; ctx.textAlign = "right";
    const lg: [string, string, number[]][] = [["참값", C.ink, []], ["쿨롱 카운팅", C.ink2, [6, 4]], ["EKF", C.edge, []]];
    let lx = A.x + A.w;
    for (let i = lg.length - 1; i >= 0; i--) {
      const [t, col, dash] = lg[i];
      ctx.fillStyle = C.mid; ctx.fillText(t, lx, A.y - 8);
      lx -= ctx.measureText(t).width + 6;
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(lx - 16, A.y - 12); ctx.lineTo(lx, A.y - 12); ctx.stroke(); ctx.setLineDash([]);
      lx -= 26;
    }
    // ── 오차
    let m = 2;
    for (let k = 0; k < T; k += 10) m = Math.max(m, Math.abs(sim.cc[k] - sim.socTrue[k]) * 100, Math.abs(sim.ekf[k] - sim.socTrue[k]) * 100);
    m *= 1.1;
    // 부호 있는 제곱근 축: 쿨롱 카운팅의 큰 오차와 EKF의 작은 오차를 한 그림에서 함께 읽는다
    const ss = (v: number) => Math.sign(v) * Math.sqrt(Math.abs(v));
    const YB = (v: number) => B.y + B.h / 2 - (ss(v) / Math.sqrt(m)) * (B.h / 2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(B.x + 0.5, B.y + 0.5, B.w - 1, B.h - 1);
    ctx.strokeStyle = C.mid; ctx.beginPath(); ctx.moveTo(B.x, YB(0)); ctx.lineTo(B.x + B.w, YB(0)); ctx.stroke();
    ctx.font = mono; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    const ticks = [1, 5, 20, 40].filter((v) => v < m * 0.95);
    for (const v of [...ticks.map((x) => -x), 0, ...ticks]) {
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(B.x, YB(v)); ctx.lineTo(B.x + B.w, YB(v)); ctx.stroke();
      ctx.fillText((v > 0 ? "+" : "") + v, B.x - 5, YB(v) + 4);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(B.x, B.y, B.w, B.h); ctx.clip();
    // EKF ±2σ 띠(필터 자신의 불확실성)
    ctx.fillStyle = alpha(C.edge, 0.18); ctx.beginPath();
    for (let k = 0; k < T; k += step) { const e = (sim.ekf[k] - sim.socTrue[k]) * 100, s = 2 * sim.sd[k] * 100; (k ? ctx.lineTo(X(k), YB(e + s)) : ctx.moveTo(X(k), YB(e + s))); }
    for (let k = T - 1; k >= 0; k -= step) { const e = (sim.ekf[k] - sim.socTrue[k]) * 100, s = 2 * sim.sd[k] * 100; ctx.lineTo(X(k), YB(e - s)); }
    ctx.closePath(); ctx.fill();
    const errOf = (a: Float64Array) => Float64Array.from(a, (v, k) => v - sim.socTrue[k]);
    curve(errOf(sim.cc), YB, C.ink2, 2, [6, 4]);
    curve(errOf(sim.ekf), YB, C.edge, 2);
    ctx.restore();
    ctx.textAlign = "left"; ctx.font = sans; ctx.fillStyle = C.mid;
    ctx.fillText("추정 오차 (%p, 제곱근 축) · EKF ±2σ", B.x, B.y - 8);
    ctx.font = mono; ctx.textAlign = "center";
    for (const t of [0, 30, 60, 90]) ctx.fillText(String(t), Math.max(X(t * 60), B.x + 8), B.y + B.h + 15);
    ctx.textAlign = "right"; ctx.fillText("120 min", B.x + B.w, B.y + B.h + 15);
    ctx.textAlign = "left";
  }

  function update() {
    sim = simulate(inp, { offsetA: offset, soc0Est: SOC_TRUE0 + err0 / 100, kappa, nominalQ: nominal });
    const e = (a: Float64Array, k: number) => (a[k] - sim.socTrue[k]) * 100;
    const s = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(2);
    let late = 0;
    for (let k = 600; k < T; k++) late = Math.max(late, Math.abs(e(sim.ekf, k)));
    ro.set("m120", `120분 CC ${s(e(sim.cc, T - 1))} · EKF ${s(e(sim.ekf, T - 1))} %p`, "hot");
    ro.set("late", `EKF 10분 뒤 최대 ${late.toFixed(2)} %p`, late < 1 ? "ok" : "");
    pageState.set("socFilter", { offset_A: offset, soc0_err_pp: err0, kappa, nominal_capacity: nominal, cc_err_120min_pp: e(sim.cc, T - 1), ekf_err_120min_pp: e(sim.ekf, T - 1), ekf_max_abs_err_after_10min_pp: late }, root);
    draw();
  }

  const pOff = param({ label: "offset", min: -0.1, max: 0.1, step: 0.01, value: offset, fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} A`, aria: "전류 센서 offset (암페어)", onInput: (v) => { offset = v; update(); } });
  const pErr = param({ label: "초기 오차", min: -40, max: 8, step: 1, value: err0, fmt: (v) => `${v >= 0 ? "+" : ""}${v} %p`, aria: "추정기의 초기 SOC 오차 (퍼센트포인트)", onInput: (v) => { err0 = v; update(); } });
  const pK = param({ label: "OCV 기울기", min: 0.02, max: 1, step: 0.02, value: kappa, fmt: (v) => `${(v * (0.55 + 0.08 / 0.52) * 10).toFixed(2)} mV/%p`, aria: "OCV 곡선의 기울기(SOC 50%에서, 최댓값이 장의 곡선)", onInput: (v) => { kappa = v; update(); } });
  for (const p of [pOff, pErr, pK]) p.row.style.gridTemplateColumns = "6em minmax(0, 1fr) 7.2em";
  const grid = el("div", "param-grid");
  grid.append(pOff.row, pErr.row, pK.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" checked /> 공칭 용량 3.0 Ah 가정(실제 2.7)`;
  sw.querySelector("input")!.addEventListener("change", (ev) => { nominal = (ev.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);
  f.onResize(draw);
  update();
}
