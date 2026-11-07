// 선점과 체크포인트 간격 (data/widgets.toml: preemption-checkpoint).
// 2.5장 "checkpoint 주기"의 모형: 학습 W = 10시간, checkpoint 주기 τ, 저장 비용 c, 선점 사이 평균 시간 M(지수분포), 재시작 비용 R = 0.1시간.
// 본문 Monte Carlo(wall_time_h)와 같은 규칙에서 기대 총 소요 시간은 닫힌 꼴로 계산된다. 길이 s = (그 구간의 작업 + c)인 구간 하나를
// 끝내는 데 드는 기대 시간은, 선점이 기억 없는(지수) 분포이므로 E = (M + R)(e^{s/M} − 1)이고, 총 소요는 구간들의 합이다.
// 1차 근사 최적 주기 τ* ≈ √(2cM)(본문 식)과 곡선의 실제 최소점을 함께 보인다. 위 띠는 같은 규칙으로 뽑은 한 번의 실행 예다.
import { frame, param, readouts, el, C, rng } from "../frame";

export const WORK = 10, RESTART = 0.1;

/** 본문 wall_time_h와 같은 순서로 구간을 나눈다(마지막 구간은 남은 작업만큼). */
export function segments(tau: number, work = WORK) {
  const seg: number[] = [];
  let done = 0;
  while (done < work) { const s = Math.min(tau, work - done); seg.push(s); done += s; }
  return seg;
}
/** 기대 총 소요 시간(시간) */
export function expectedTotal(tau: number, c: number, M: number, R = RESTART, work = WORK) {
  return segments(tau, work).reduce((t, s) => t + (M + R) * Math.expm1((s + c) / M), 0);
}
/** 로그 격자에서 찾은 실제 최소점 */
export function bestTau(c: number, M: number) {
  let bt = 0, bv = Infinity;
  for (let i = 0; i <= 600; i++) {
    const tau = 0.02 * Math.pow(WORK / 0.02, i / 600), v = expectedTotal(tau, c, M);
    if (v < bv) { bv = v; bt = tau; }
  }
  return { tau: bt, total: bv };
}
/** 한 번의 실행: 본문 코드와 같은 규칙으로 선점을 뽑아 사건 목록을 만든다 */
function sampleRun(tau: number, c: number, M: number, seed = 7) {
  const r = rng(seed);
  const ev: { kind: "work" | "ckpt" | "lost" | "restart"; len: number }[] = [];
  let done = 0, guard = 0;
  while (done < WORK && guard++ < 5000) {
    const seg = Math.min(tau, WORK - done);
    const fail = -M * Math.log(Math.max(r.u(), 1e-12));
    if (fail >= seg + c) { ev.push({ kind: "work", len: seg }, { kind: "ckpt", len: c }); done += seg; }
    else { ev.push({ kind: "lost", len: fail }, { kind: "restart", len: RESTART }); }
  }
  return ev;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "선점과 체크포인트 간격", "manipulate", narrow ? 1.05 : 0.56, "checkpoint 주기에 따른 기대 총 소요 시간 곡선과 근사 최적 주기, 그리고 한 번의 실행 예");
  let tau = 4, cMin = 3, M = 8;
  const T0 = 0.05, T1 = WORK;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const c = cMin / 60;
    const L = narrow ? 40 : 50, R = 16, stripTop = 26, stripH = 22;
    const top = stripTop + stripH + 44, bottom = H - 30;
    ctx.textBaseline = "alphabetic";

    // 위: 한 번의 실행 예
    const ev = sampleRun(tau, c, M);
    const total = ev.reduce((t, e) => t + e.len, 0);
    const span = Math.max(total, WORK * 1.05);
    const SX = (t: number) => L + (t / span) * (W - L - R);
    let t = 0;
    for (const e of ev) {
      const x0 = SX(t), x1 = SX(t + e.len);
      ctx.fillStyle = e.kind === "work" ? C.ink2 : e.kind === "ckpt" ? C.ink : e.kind === "lost" ? C.edge : C.mid;
      ctx.globalAlpha = e.kind === "lost" ? 0.75 : 1;
      ctx.fillRect(x0, stripTop, Math.max(x1 - x0, e.kind === "ckpt" ? 1.5 : 0.5), stripH);
      ctx.globalAlpha = 1;
      t += e.len;
    }
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText(`한 번의 실행 예: ${total.toFixed(1)} h`, L, stripTop - 8);
    const legend: [string, string][] = [["작업", C.ink2], ["저장", C.ink], ["잃은 작업", C.edge], ["재시작", C.mid]];
    ctx.font = "11px Pretendard Variable, sans-serif";
    let lx = L;
    const ly = stripTop + stripH + 16;
    for (const [name, col] of legend) { ctx.fillStyle = col; ctx.fillRect(lx, ly - 8, 9, 9); ctx.fillStyle = C.mid; ctx.fillText(name, lx + 13, ly); lx += ctx.measureText(name).width + 26; }

    // 아래: 기대 총 소요 시간 곡선(τ는 로그 눈금)
    const X = (v: number) => L + (Math.log(v / T0) / Math.log(T1 / T0)) * (W - L - R);
    const N = 240, taus = Array.from({ length: N + 1 }, (_, i) => T0 * Math.pow(T1 / T0, i / N));
    const vals = taus.map((v) => expectedTotal(v, c, M));
    const best = bestTau(c, M);
    const ymin = WORK * 0.95, ymax = Math.min(Math.max(...vals), WORK * 3) * 1.04;
    const Y = (v: number) => bottom - ((Math.min(v, ymax) - ymin) / (ymax - ymin)) * (bottom - top);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.strokeRect(L, top, W - L - R, bottom - top);
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("기대 총 소요 시간 [h]", L, top - 8);
    ctx.font = "11px JetBrains Mono, monospace";
    for (const tk of [0.1, 0.25, 0.5, 1, 2, 5, 10]) {
      if (narrow && (tk === 0.25 || tk === 5)) continue;
      const s = String(tk);
      ctx.fillText(s, X(tk) - ctx.measureText(s).width / 2, bottom + 14);
    }
    ctx.fillText("τ [h]", W - R - 34, bottom + 26);
    const ystep = ymax - ymin > 20 ? 10 : ymax - ymin > 8 ? 5 : 2;
    for (let v = Math.ceil(ymin / ystep) * ystep; v < ymax; v += ystep) { ctx.fillText(String(v), L - 24, Y(v) + 4); }
    // 작업량 W(선점도 저장도 없을 때)
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.mid;
    ctx.beginPath(); ctx.moveTo(L, Y(WORK)); ctx.lineTo(W - R, Y(WORK)); ctx.stroke();
    // 근사 최적 √(2cM)
    const approx = Math.sqrt(2 * c * M);
    if (approx >= T0 && approx <= T1) { ctx.strokeStyle = C.ink2; ctx.beginPath(); ctx.moveTo(X(approx), top); ctx.lineTo(X(approx), bottom); ctx.stroke(); }
    ctx.setLineDash([]);
    if (approx >= T0 && approx <= T1) { ctx.fillStyle = C.ink2; ctx.font = "600 12px Pretendard Variable, sans-serif"; const s = "√(2cM)"; ctx.fillText(s, Math.min(X(approx) + 5, W - R - ctx.measureText(s).width - 2), top + 14); }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2;
    ctx.save(); ctx.beginPath(); ctx.rect(L, top, W - L - R, bottom - top); ctx.clip();
    ctx.beginPath(); taus.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(vals[i])) : ctx.moveTo(X(v), Y(vals[i])))); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = C.good;
    ctx.beginPath(); ctx.arc(X(best.tau), Y(best.total), 4.5, 0, 7); ctx.fill();
    // 고른 τ
    const cur = expectedTotal(tau, c, M);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(X(tau), top); ctx.lineTo(X(tau), bottom); ctx.stroke();
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(X(tau), Y(cur), 6, 0, 7); ctx.fill();
  }

  function update() {
    const c = cMin / 60;
    const best = bestTau(c, M), cur = expectedTotal(tau, c, M);
    ro.set("cur", `τ = ${tau.toFixed(2)} h → 기대 ${cur.toFixed(2)} h`, "hot");
    ro.set("approx", `√(2cM) = ${Math.sqrt(2 * c * M).toFixed(2)} h`);
    ro.set("best", `곡선 최소 ${best.tau.toFixed(2)} h · ${best.total.toFixed(2)} h`, "ok");
    ro.set("none", `checkpoint 없음 ${expectedTotal(WORK, c, M).toFixed(2)} h`);
    draw();
  }

  const pt = param({ label: "τ", min: Math.log10(T0), max: 1, step: 0.01, value: Math.log10(tau), fmt: (v) => `${Math.pow(10, v).toFixed(2)} h`, aria: "checkpoint 주기 τ (시간, 로그 눈금)", onInput: (v) => { tau = Math.pow(10, v); update(); } });
  const pc = param({ label: "c", min: 0.5, max: 30, step: 0.5, value: cMin, fmt: (v) => `${v} 분`, aria: "checkpoint 저장 한 번의 비용 c (분)", onInput: (v) => { cMin = v; update(); } });
  const pm = param({ label: "M", min: 1, max: 48, step: 1, value: M, fmt: (v) => `${v} h`, aria: "선점 사이의 평균 시간 M (시간)", onInput: (v) => { M = v; update(); } });
  for (const p of [pt, pc, pm]) p.row.style.gridTemplateColumns = "1.6em minmax(0, 1fr) 4.4em";
  const grid = el("div", "param-grid");
  grid.append(pt.row, pc.row, pm.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);

  // 그림 위에서 τ 끌기
  let dragging = false;
  const fromEvent = (e: PointerEvent) => {
    const r = f.canvas.getBoundingClientRect();
    const L = narrow ? 40 : 50;
    const u = Math.min(1, Math.max(0, (e.clientX - r.left - L) / (f.width - L - 16)));
    tau = T0 * Math.pow(T1 / T0, u);
    pt.set(Math.round(Math.log10(tau) * 100) / 100);
    update();
  };
  f.canvas.addEventListener("pointerdown", (e) => {
    if (e.clientY - f.canvas.getBoundingClientRect().top < 80) return;   // 위의 실행 띠는 시간축이 다르다
    dragging = true; f.canvas.setPointerCapture(e.pointerId); fromEvent(e);
  });
  f.canvas.addEventListener("pointermove", (e) => { if (dragging) fromEvent(e); });
  f.canvas.addEventListener("pointerup", () => (dragging = false));
  f.canvas.style.cursor = "ew-resize";
  f.onResize(draw);
  update();
}
