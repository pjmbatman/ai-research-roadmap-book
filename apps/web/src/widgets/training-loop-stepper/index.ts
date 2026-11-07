// 학습 루프 단계 재생기 (data/widgets.toml: training-loop-stepper).
// 선형 모델 ŷ = w·x + b를 평균제곱오차로 경사하강한다. 한 반복을 forward → loss → backward → step 네 단계로 나눠 보인다.
// 왼쪽 위: 데이터와 현재 직선(예측·잔차). 오른쪽 위: 파라미터 공간 (w, b)의 손실 등고선, 지나온 경로와 이번 갱신 −η·grad.
// 아래: 반복에 따른 손실(로그 축). 학습률 η가 안정 한계 2/λ_max(헤세 행렬의 최대 고윳값)를 넘으면 발산한다.
// zero_grad를 끄면 .grad가 반복마다 누적되어(PyTorch에서 zero_grad를 빼먹은 것과 같다) 손실이 진동한다.
import { frame, param, presets, readouts, el, C, rng } from "../frame";
import { pageState } from "../../client/store";

const NPT = 10;
const XS = Array.from({ length: NPT }, (_, i) => 0.5 + (2.5 * i) / (NPT - 1));
const YS = (() => { const r = rng(31337); return XS.map((x) => 1.2 * x - 0.8 + 0.25 * r.normal()); })();
const MX = XS.reduce((a, b) => a + b, 0) / NPT, MXX = XS.reduce((a, b) => a + b * b, 0) / NPT;
// 헤세 행렬 H = 2 [[E x², E x], [E x, 1]]의 최대 고윳값
const LMAX = (MXX + 1) + Math.sqrt((MXX - 1) ** 2 + 4 * MX * MX);
const ETA_MAX = 2 / LMAX;
// 최소제곱해 θ* = (w*, b*)
const MY = YS.reduce((a, b) => a + b, 0) / NPT, MXY = XS.reduce((a, x, i) => a + x * YS[i], 0) / NPT;
const WSTAR = (MXY - MX * MY) / (MXX - MX * MX), BSTAR = MY - WSTAR * MX;
const TH0: [number, number] = [-0.6, 1.8];
const WR: [number, number] = [-1.6, 3.2], BR: [number, number] = [-4.2, 3.0];
const MAX_T = 400;

type Phase = "forward" | "loss" | "backward" | "step";
const PHASES: Phase[] = ["forward", "loss", "backward", "step"];
const CAP: Record<Phase, string> = {
  forward: "forward: 현재 파라미터로 예측한다. <code>pred = model(x)</code>",
  loss: "loss: 잔차를 제곱해 평균한다. <code>loss = loss_fn(pred, y)</code>",
  backward: "backward: 손실의 기울기 ∇θL을 .grad에 더한다. <code>loss.backward()</code>",
  step: "step: θ ← θ − η·grad로 갱신하고 .grad를 비운다. <code>opt.step(); opt.zero_grad()</code>",
};
const CAP_NOZERO = "step: θ ← θ − η·grad로 갱신한다. zero_grad가 없어 .grad가 다음 반복으로 넘어간다. <code>opt.step()</code>";

const lossAt = (w: number, b: number) => { let s = 0; for (let i = 0; i < NPT; i++) s += (w * XS[i] + b - YS[i]) ** 2; return s / NPT; };
const gradAt = (w: number, b: number): [number, number] => {
  let gw = 0, gb = 0;
  for (let i = 0; i < NPT; i++) { const e = w * XS[i] + b - YS[i]; gw += e * XS[i]; gb += e; }
  return [(2 * gw) / NPT, (2 * gb) / NPT];
};

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "학습 루프 단계 재생기", "stepper", narrow ? 1.05 : 0.74, "데이터와 현재 직선, 파라미터 공간의 손실 등고선과 경로, 반복별 손실 곡선");
  let eta = 0.1, zeroGrad = true;
  let theta: [number, number] = [...TH0];
  let gbuf: [number, number] = [0, 0]; // 이번 반복의 backward 전 .grad
  let phase = 0, t = 0;
  let hist: number[] = [lossAt(...TH0)];
  let traj: [number, number][] = [[...TH0]];
  let playing = 0;

  const derived = () => {
    const L = lossAt(...theta);
    const g = gradAt(...theta);
    const grad: [number, number] = [gbuf[0] + g[0], gbuf[1] + g[1]];
    const next: [number, number] = [theta[0] - eta * grad[0], theta[1] - eta * grad[1]];
    return { L, g, grad, next };
  };
  const diverged = () => !Number.isFinite(hist[hist.length - 1]) || hist[hist.length - 1] > 1e6;

  /** step 단계 다음으로 넘어가며 갱신을 확정한다. */
  function commit() {
    const d = derived();
    theta = d.next;
    gbuf = zeroGrad ? [0, 0] : d.grad;
    t++;
    hist.push(lossAt(...theta));
    traj.push([...theta]);
  }
  function advance() {
    if (diverged() || t >= MAX_T) return false;
    if (phase < 3) phase++;
    else { commit(); phase = 0; }
    return true;
  }

  // 등고선(마칭 스퀘어): 손실 log 값의 등간격 수준
  const GN = 64;
  const grid = new Float64Array((GN + 1) * (GN + 1));
  for (let j = 0; j <= GN; j++) for (let i = 0; i <= GN; i++) {
    const w = WR[0] + ((WR[1] - WR[0]) * i) / GN, b = BR[1] - ((BR[1] - BR[0]) * j) / GN;
    grid[j * (GN + 1) + i] = Math.log(lossAt(w, b));
  }
  const LV = Array.from({ length: 12 }, (_, k) => Math.log(lossAt(WSTAR, BSTAR)) + 0.35 + 0.55 * k);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const pad = 12, gap = 16, head = 22;
    const stripH = Math.max(76, Math.round(H * 0.24));
    const topH = H - stripH - head * 2 - pad;
    const half = (W - 2 * pad - gap) / 2;
    const d = derived();
    const ph = PHASES[phase];
    ctx.font = "600 12px Pretendard Variable, sans-serif";

    // 1) 데이터와 직선
    const A = { x: pad, y: head, w: half, h: topH };
    const X = (x: number) => A.x + 6 + ((x - 0.2) / 3.1) * (A.w - 12);
    const Y = (y: number) => A.y + A.h - ((y + 1.2) / 5.0) * A.h;
    ctx.save(); ctx.beginPath(); ctx.rect(A.x, A.y, A.w, A.h); ctx.clip();
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(A.x + 0.5, A.y + 0.5, A.w - 1, A.h - 1);
    const showNext = ph === "step";
    const line = (w: number, b: number, col: string, lw: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(X(0.2), Y(w * 0.2 + b)); ctx.lineTo(X(3.3), Y(w * 3.3 + b)); ctx.stroke(); ctx.setLineDash([]);
    };
    line(theta[0], theta[1], showNext ? C.mid : C.ink, showNext ? 1.2 : 2, showNext ? [4, 4] : []);
    if (showNext) line(d.next[0], d.next[1], C.edge, 2.4);
    for (let i = 0; i < NPT; i++) {
      const yh = theta[0] * XS[i] + theta[1];
      if (ph === "loss") { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(XS[i]), Y(YS[i])); ctx.lineTo(X(XS[i]), Y(yh)); ctx.stroke(); }
      ctx.fillStyle = C.ink2; ctx.beginPath(); ctx.arc(X(XS[i]), Y(YS[i]), 3.6, 0, 7); ctx.fill();
      if (ph === "forward") { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X(XS[i]), Y(yh), 3.6, 0, 7); ctx.fill(); }
    }
    ctx.restore();
    ctx.fillStyle = C.mid;
    ctx.fillText("데이터와 ŷ = w·x + b", A.x, A.y - 7);

    // 2) 파라미터 공간
    const B = { x: pad + half + gap, y: head, w: half, h: topH };
    const PX = (w: number) => B.x + ((w - WR[0]) / (WR[1] - WR[0])) * B.w;
    const PY = (b: number) => B.y + ((BR[1] - b) / (BR[1] - BR[0])) * B.h;
    ctx.save(); ctx.beginPath(); ctx.rect(B.x, B.y, B.w, B.h); ctx.clip();
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(B.x + 0.5, B.y + 0.5, B.w - 1, B.h - 1);
    const sx = B.w / GN, sy = B.h / GN;
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
    ctx.beginPath();
    for (const lv of LV) {
      for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) {
        const v = [grid[j * (GN + 1) + i], grid[j * (GN + 1) + i + 1], grid[(j + 1) * (GN + 1) + i + 1], grid[(j + 1) * (GN + 1) + i]];
        const pts: [number, number][] = [];
        const corner = [[0, 0], [1, 0], [1, 1], [0, 1]];
        for (let e = 0; e < 4; e++) {
          const a = v[e], c = v[(e + 1) % 4];
          if ((a - lv) * (c - lv) < 0) {
            const s = (lv - a) / (c - a);
            const [x0, y0] = corner[e], [x1, y1] = corner[(e + 1) % 4];
            pts.push([B.x + (i + x0 + (x1 - x0) * s) * sx, B.y + (j + y0 + (y1 - y0) * s) * sy]);
          }
        }
        for (let p = 0; p + 1 < pts.length; p += 2) { ctx.moveTo(...pts[p]); ctx.lineTo(...pts[p + 1]); }
      }
    }
    ctx.stroke();
    // 최솟값
    const [ow, ob] = [WSTAR, BSTAR];
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(PX(ow) - 5, PY(ob) - 5); ctx.lineTo(PX(ow) + 5, PY(ob) + 5); ctx.moveTo(PX(ow) + 5, PY(ob) - 5); ctx.lineTo(PX(ow) - 5, PY(ob) + 5); ctx.stroke();
    // 경로
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.6;
    ctx.beginPath(); traj.forEach(([w, b], k) => (k ? ctx.lineTo(PX(w), PY(b)) : ctx.moveTo(PX(w), PY(b)))); ctx.stroke();
    ctx.fillStyle = C.ink2;
    traj.slice(-40).forEach(([w, b]) => { ctx.beginPath(); ctx.arc(PX(w), PY(b), 2, 0, 7); ctx.fill(); });
    // 이번 갱신 −η·grad
    if (ph === "backward" || ph === "step") {
      const a: [number, number] = [PX(theta[0]), PY(theta[1])], c: [number, number] = [PX(d.next[0]), PY(d.next[1])];
      ctx.strokeStyle = C.edge; ctx.fillStyle = C.edge; ctx.lineWidth = 2.2;
      ctx.setLineDash(ph === "backward" ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...c); ctx.stroke(); ctx.setLineDash([]);
      const ang = Math.atan2(c[1] - a[1], c[0] - a[0]);
      if (Math.hypot(c[0] - a[0], c[1] - a[1]) > 6) {
        ctx.beginPath(); ctx.moveTo(...c);
        ctx.lineTo(c[0] - 10 * Math.cos(ang - 0.4), c[1] - 10 * Math.sin(ang - 0.4));
        ctx.lineTo(c[0] - 10 * Math.cos(ang + 0.4), c[1] - 10 * Math.sin(ang + 0.4)); ctx.fill();
      }
    }
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(PX(theta[0]), PY(theta[1]), 5, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = C.mid;
    ctx.fillText("손실 등고선 (w, b)", B.x, B.y - 7);

    // 3) 손실 곡선(로그 축)
    const S = { x: pad, y: head * 2 + topH, w: W - 2 * pad, h: stripH - 4 };
    const T = Math.max(40, t);
    const finite = hist.filter((v) => Number.isFinite(v));
    const lmin = Math.log10(Math.min(0.02, ...finite)), lmax = Math.log10(Math.max(3, ...finite.map((v) => Math.min(v, 1e6))));
    const SX = (k: number) => S.x + 6 + (k / T) * (S.w - 12);
    const SY = (v: number) => S.y + S.h - ((Math.log10(Math.min(Math.max(v, 1e-9), 1e6)) - lmin) / (lmax - lmin)) * S.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(S.x, S.y + S.h); ctx.lineTo(S.x + S.w, S.y + S.h); ctx.stroke();
    const lstar = lossAt(ow, ob);
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(S.x, SY(lstar)); ctx.lineTo(S.x + S.w, SY(lstar)); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.beginPath(); ctx.rect(S.x, S.y - 4, S.w, S.h + 8); ctx.clip();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); hist.forEach((v, k) => (k ? ctx.lineTo(SX(k), SY(v)) : ctx.moveTo(SX(k), SY(v)))); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(SX(t), SY(hist[t]), 4, 0, 7); ctx.fill();
    ctx.restore();
    ctx.fillStyle = C.mid;
    ctx.fillText("반복별 손실 L (로그 축)", S.x, S.y - 7);
  }

  function update() {
    const d = derived();
    const ph = PHASES[phase];
    phaseBtns.select(ph);
    cap.innerHTML = ph === "step" && !zeroGrad ? CAP_NOZERO : CAP[ph];
    const f2 = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(2));
    ro.set("t", `반복 ${t}`);
    ro.set("loss", diverged() ? "발산" : `L ${d.L.toFixed(4)}`, ph === "loss" || diverged() ? "hot" : "");
    ro.set("grad", `grad (${f2(d.grad[0])}, ${f2(d.grad[1])})`, ph === "backward" ? "hot" : "");
    ro.set("theta", ph === "step" ? `θ → (${f2(d.next[0])}, ${f2(d.next[1])})` : `θ (${f2(theta[0])}, ${f2(theta[1])})`, ph === "step" ? "hot" : "");
    ro.set("lim", `안정 한계 η < ${ETA_MAX.toFixed(3)}`, eta >= ETA_MAX ? "hot" : "");
    pageState.set("trainingLoop", { t, phase: ph, eta, zeroGrad, loss: d.L, theta: [...theta], grad: d.grad }, root);
    draw();
  }

  function stop() { if (playing) { clearInterval(playing); playing = 0; } play.textContent = "▶ 재생"; }
  function start() {
    play.textContent = "⏸ 정지";
    phase = 3;
    playing = window.setInterval(() => {
      if (!f.canvas.isConnected) { stop(); return; }
      const ok = advance() && advance() && advance() && advance() && !diverged() && t < MAX_T;
      update();
      if (!ok) stop();
    }, 60);
  }

  const phaseBtns = presets<Phase>(PHASES.map((p) => ({ key: p, label: p })), (k) => {
    stop();
    phase = PHASES.indexOf(k);
    update();
  }, "forward");
  phaseBtns.row.classList.add("segmented");
  if (narrow) phaseBtns.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 9px"));
  const cap = el("p", "wcap");
  const pe = param({ label: "η", min: 0.01, max: 0.25, step: 0.005, value: eta, fmt: (v) => v.toFixed(3), aria: "학습률 η", onInput: (v) => { eta = v; update(); } });
  const pgrid = el("div", "param-grid");
  pgrid.append(pe.row);
  const tools = el("div", "wframe-row");
  const next = el("button", "preset");
  next.type = "button";
  next.textContent = "다음 단계 →";
  next.addEventListener("click", () => { stop(); advance(); update(); });
  const play = el("button", "preset");
  play.type = "button";
  play.textContent = "▶ 재생";
  play.addEventListener("click", () => (playing ? stop() : start()));
  const zg = el("label", "toggle");
  zg.innerHTML = `<input type="checkbox" checked /> zero_grad`;
  zg.querySelector("input")!.addEventListener("change", (e) => { zeroGrad = (e.target as HTMLInputElement).checked; update(); });
  tools.append(next, play, zg);
  const ro = readouts();
  f.panel.append(phaseBtns.row, cap, pgrid, tools, ro.row);
  f.onResize(draw);
  update();
}
