// 1차원 칼만 필터 (data/widgets.toml: kalman-1d). 16.7장의 local level 모델(F = H = 1)과 같은 설정이다.
// 숨은 상태는 random walk(상태 잡음 sd 0.05), 관측 잡음 sd 0.5, 사전 m₀ = 0, P₀ = 10. 데이터는 시드 고정으로 한 번 만든다.
// 필터에 넣는 q, r만 바꾼다(데이터는 그대로). 위: 관측·참 상태·필터 평균과 95% 구간. 아래: 칼만 이득 K_t.
// 읽음값: 정상 상태 이득 K∞, 참 상태 대비 RMSE, 95% 구간 포함률, 정규화 혁신 제곱(NIS) 평균(모델이 맞으면 1 근처).
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";

export const T = 150, Q_TRUE = 0.05, R_TRUE = 0.5, P0 = 10;

/** 참 상태와 관측: x_t = x_{t−1} + w_t, y_t = x_t + v_t */
export function makeData(seed = 7) {
  const r = rng(seed);
  const x: number[] = [], y: number[] = [];
  let s = 0;
  for (let t = 0; t < T; t++) { s += Q_TRUE * r.normal(); x.push(s); y.push(s + R_TRUE * r.normal()); }
  return { x, y };
}

/** 스칼라 칼만 필터. q, r은 분산이다. */
export function kalman(y: number[], q: number, r: number) {
  let m = 0, P = P0;
  const mean: number[] = [], sd: number[] = [], K: number[] = [], nis: number[] = [];
  for (const yt of y) {
    const Pp = P + q;                  // 예측
    const S = Pp + r;                  // 혁신 분산
    const k = Pp / S;                  // 칼만 이득
    const e = yt - m;                  // 혁신
    nis.push((e * e) / S);
    m = m + k * e; P = (1 - k) * Pp;   // 갱신
    mean.push(m); sd.push(Math.sqrt(P)); K.push(k);
  }
  return { mean, sd, K, nis };
}
/** 정상 상태 이득: 예측 분산 P⁻ = (q + √(q² + 4qr))/2, K∞ = P⁻/(P⁻ + r) */
export const kInf = (q: number, r: number) => { const Pp = (q + Math.sqrt(q * q + 4 * q * r)) / 2; return Pp / (Pp + r); };

export function metrics(x: number[], y: number[], q: number, r: number) {
  const k = kalman(y, q, r);
  let se = 0, seObs = 0, cover = 0, nis = 0;
  for (let t = 0; t < x.length; t++) {
    se += (k.mean[t] - x[t]) ** 2; seObs += (y[t] - x[t]) ** 2;
    if (Math.abs(k.mean[t] - x[t]) < 1.96 * k.sd[t]) cover++;
    nis += k.nis[t];
  }
  const n = x.length;
  return { ...k, rmse: Math.sqrt(se / n), rmseObs: Math.sqrt(seObs / n), cover: cover / n, nisMean: nis / n, kInf: kInf(q, r) };
}

const PRESETS = {
  match: { label: "참값", q: 0.05, r: 0.5 },
  qsmall: { label: "q 과소", q: 0.005, r: 0.5 },
  qbig: { label: "q 과대", q: 0.5, r: 0.5 },
  rbig: { label: "r 과대", q: 0.05, r: 5 },
};
type Key = keyof typeof PRESETS;

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "1D 칼만 필터", "manipulate", narrow ? 0.95 : 0.58, "잡음 섞인 관측, 참 상태, 필터 평균과 95% 구간, 그리고 시간에 따른 칼만 이득");
  const data = makeData();
  let qs = 0.05, rs = 0.5;
  let res = metrics(data.x, data.y, qs * qs, rs * rs);
  const ymin = Math.min(...data.y) - 0.2, ymax = Math.max(...data.y) + 0.2;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const left = 34, right = W - 10, top = 10;
    const kH = Math.round(H * 0.24), split = H - kH - 34;
    const X = (t: number) => left + (t / (T - 1)) * (right - left);
    const Y = (v: number) => top + ((ymax - v) / (ymax - ymin)) * (split - top);
    ctx.font = "11px JetBrains Mono, monospace";
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(left, top, right - left, split - top);
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (let v = Math.ceil(ymin); v <= Math.floor(ymax); v++) {
      ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(right, Y(v)); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(String(v), left - 5, Y(v) + 4);
    }
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, split - top); ctx.clip();
    // 95% 구간
    ctx.fillStyle = C.fill;
    ctx.beginPath();
    for (let t = 0; t < T; t++) { const v = res.mean[t] + 1.96 * res.sd[t]; t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)); }
    for (let t = T - 1; t >= 0; t--) ctx.lineTo(X(t), Y(res.mean[t] - 1.96 * res.sd[t]));
    ctx.closePath(); ctx.fill();
    // 관측
    ctx.fillStyle = alpha(C.mid, 0.7);
    for (let t = 0; t < T; t++) { ctx.beginPath(); ctx.arc(X(t), Y(data.y[t]), 1.8, 0, 7); ctx.fill(); }
    // 참 상태
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
    ctx.beginPath(); data.x.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)))); ctx.stroke();
    ctx.setLineDash([]);
    // 필터 평균
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    ctx.beginPath(); res.mean.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)))); ctx.stroke();
    ctx.restore();
    // 범례
    const items: [string, (x: number, y: number) => void][] = [
      ["관측", (x, y) => { ctx.fillStyle = C.mid; ctx.beginPath(); ctx.arc(x + 7, y, 2.4, 0, 7); ctx.fill(); }],
      ["참 상태", (x, y) => { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); ctx.setLineDash([]); }],
      ["필터 m ± 1.96√P", (x, y) => { ctx.fillStyle = C.fill; ctx.fillRect(x, y - 5, 14, 10); ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); }],
    ];
    let lx = left + 6;
    const ly = top + 12;
    const wsum = items.reduce((s, [t]) => s + ctx.measureText(t).width + 26, 0);
    ctx.fillStyle = alpha(C.bg, 0.85); ctx.fillRect(left + 1, top + 1, Math.min(wsum, right - left - 2), 21);
    for (const [txt, mark] of items) {
      mark(lx, ly - 4);
      ctx.fillStyle = C.ink; ctx.fillText(txt, lx + 18, ly);
      lx += ctx.measureText(txt).width + 26;
    }
    ctx.fillStyle = C.mid; ctx.textAlign = "center";
    ctx.fillText("t", X(T / 2), split + 13);
    ctx.textAlign = "start";
    // 아래: 칼만 이득
    const kt = split + 22, kb = H - 8;
    const KY = (k: number) => kb - k * (kb - kt);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(left, kt, right - left, kb - kt);
    ctx.fillStyle = C.mid; ctx.textAlign = "right";
    ctx.fillText("1", left - 5, KY(1) + 8); ctx.fillText("0", left - 5, KY(0));
    ctx.textAlign = "start";
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(left, KY(res.kInf)); ctx.lineTo(right, KY(res.kInf)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); res.K.forEach((k, t) => (t ? ctx.lineTo(X(t), KY(k)) : ctx.moveTo(X(t), KY(k)))); ctx.stroke();
    ctx.fillStyle = C.ink2;
    const kl = `K_t → ${res.kInf.toFixed(3)}`;
    const ky = res.kInf > 0.6 ? KY(res.kInf) + 14 : KY(res.kInf) - 5;
    ctx.fillText(kl, right - ctx.measureText(kl).width - 6, Math.max(kt + 12, Math.min(kb - 3, ky)));
  }

  function update() {
    res = metrics(data.x, data.y, qs * qs, rs * rs);
    ro.set("k", `K∞ = ${res.kInf.toFixed(3)}`);
    ro.set("rmse", `RMSE ${res.rmse.toFixed(3)} (관측 ${res.rmseObs.toFixed(3)})`);
    ro.set("cov", `95% 구간 포함률 ${res.cover.toFixed(2)}`, res.cover < 0.85 ? "hot" : res.cover > 0.9 ? "ok" : "");
    const bad = res.nisMean > 1.5 || res.nisMean < 0.6;
    ro.set("nis", `NIS 평균 ${res.nisMean.toFixed(2)}`, bad ? "hot" : "ok");
    draw();
  }

  const lg = (v: number) => Math.log10(v);
  const pq = param({ label: "q sd", min: lg(0.002), max: lg(1), step: 0.01, value: lg(qs), fmt: (v) => (10 ** v).toPrecision(2), aria: "필터가 가정하는 상태 잡음 표준편차(로그 눈금)", onInput: (v) => { qs = 10 ** v; pre.select(null); update(); } });
  const pr = param({ label: "r sd", min: lg(0.05), max: lg(10), step: 0.01, value: lg(rs), fmt: (v) => (10 ** v).toPrecision(2), aria: "필터가 가정하는 관측 잡음 표준편차(로그 눈금)", onInput: (v) => { rs = 10 ** v; pre.select(null); update(); } });
  const pre = presets<Key>((Object.keys(PRESETS) as Key[]).map((k) => ({ key: k, label: PRESETS[k].label })), (k) => {
    qs = PRESETS[k].q; rs = PRESETS[k].r; pq.set(lg(qs)); pr.set(lg(rs)); update();
  }, "match");
  const grid = el("div", "param-grid");
  grid.append(pq.row, pr.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
