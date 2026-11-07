// 분자동역학 에너지 보존 (data/widgets.toml: md-energy-conservation).
// 2차원 Lennard-Jones 원자 7개(육각 무리, LJ 단위 ε = σ = m = 1)를 T = 16 동안 적분하고 총에너지 E = K + U의 시계열을 그린다.
// 적분기: velocity Verlet(반 스텝 속도 → 위치 → 새 힘 → 반 스텝 속도) 또는 explicit Euler(옛 힘·옛 속도로 한 번에).
// 비보존력 스위치를 켜면 참 LJ 힘에 회전형 성분 γ ẑ × (rᵢ − r_cm)(γ = 0.05)을 더한다. 이 힘장은 어떤 에너지의 gradient도 아니다.
// 보존력 + Verlet이면 표류가 Δt²로 줄고, 비보존력이면 Δt를 줄여도 에너지가 체계적으로 늘며, Euler는 작은 Δt에서도 에너지가 샌다.
import { frame, param, presets, readouts, el, C, rng, reducedMotion } from "../frame";

export const NAT = 7, T_END = 16, GAMMA = 0.05, R_BLOW = 4;
const RE = 2 ** (1 / 6);
export type Integ = "verlet" | "euler";

/** 처음 배치(시드 고정): 중심 + 육각 고리, 위치 흔들림 0.03, 속도 표준편차 0.15(총운동량 0) */
export function initial() {
  const r = rng(2803);
  const R = new Float64Array(2 * NAT), V = new Float64Array(2 * NAT);
  for (let i = 1; i < NAT; i++) { const a = ((i - 1) * Math.PI) / 3; R[2 * i] = RE * Math.cos(a); R[2 * i + 1] = RE * Math.sin(a); }
  for (let k = 0; k < 2 * NAT; k++) { R[k] += 0.03 * r.normal(); V[k] = 0.15 * r.normal(); }
  for (const c of [0, 1]) { let m = 0; for (let i = 0; i < NAT; i++) m += V[2 * i + c]; m /= NAT; for (let i = 0; i < NAT; i++) V[2 * i + c] -= m; }
  return { R, V };
}

export function ljEnergy(R: Float64Array) {
  let U = 0;
  for (let i = 0; i < NAT; i++) for (let j = i + 1; j < NAT; j++) {
    const r2 = (R[2 * i] - R[2 * j]) ** 2 + (R[2 * i + 1] - R[2 * j + 1]) ** 2, i6 = 1 / r2 ** 3;
    U += 4 * (i6 * i6 - i6);
  }
  return U;
}
/** F = −∇U (+ 비보존 회전 성분). extra에는 더한 성분의 제곱합, lj에는 참 힘의 제곱합을 돌려준다. */
function forces(R: Float64Array, F: Float64Array, gamma: number) {
  F.fill(0);
  for (let i = 0; i < NAT; i++) for (let j = i + 1; j < NAT; j++) {
    const dx = R[2 * i] - R[2 * j], dy = R[2 * i + 1] - R[2 * j + 1], r2 = dx * dx + dy * dy, i6 = 1 / r2 ** 3;
    const c = (24 * (2 * i6 * i6 - i6)) / r2;
    F[2 * i] += c * dx; F[2 * i + 1] += c * dy; F[2 * j] -= c * dx; F[2 * j + 1] -= c * dy;
  }
  let lj = 0; for (let k = 0; k < 2 * NAT; k++) lj += F[k] * F[k];
  let extra = 0;
  if (gamma) {
    let mx = 0, my = 0; for (let i = 0; i < NAT; i++) { mx += R[2 * i] / NAT; my += R[2 * i + 1] / NAT; }
    for (let i = 0; i < NAT; i++) {
      const fx = -gamma * (R[2 * i + 1] - my), fy = gamma * (R[2 * i] - mx);
      F[2 * i] += fx; F[2 * i + 1] += fy; extra += fx * fx + fy * fy;
    }
  }
  return { lj, extra };
}

/** 적분. E·t·frames는 그림용 표본(최대 약 400장), maxDrift·endDrift는 모든 스텝에서 잰 max|E − E₀|와 E(끝) − E₀. */
export function simulate(integ: Integ, dt: number, nonCons: boolean) {
  const { R, V } = initial();
  const F = new Float64Array(2 * NAT), g = nonCons ? GAMMA : 0;
  const steps = Math.round(T_END / dt), every = Math.max(1, Math.ceil(steps / 400));
  const ke = () => { let k = 0; for (let i = 0; i < 2 * NAT; i++) k += 0.5 * V[i] * V[i]; return k; };
  let acc = forces(R, F, g), sumLj = acc.lj, sumEx = acc.extra;
  const E0 = ke() + ljEnergy(R);
  // 그림용 표본: 저장 구간마다 E의 최소·최대(봉투)를 함께 둔다(표본만 찍으면 빠른 진동이 엇갈려 보인다)
  const E = [E0], lo = [E0], hi = [E0], t = [0], frames: Float64Array[] = [R.slice()];
  let bLo = Infinity, bHi = -Infinity;
  const save = (e: number, tt: number) => { E.push(e); lo.push(bLo); hi.push(bHi); t.push(tt); frames.push(R.slice()); bLo = Infinity; bHi = -Infinity; };
  let blown = false, n = 0, maxDrift = 0, last = E0;
  for (n = 1; n <= steps; n++) {
    if (integ === "verlet") {
      for (let k = 0; k < 2 * NAT; k++) { V[k] += 0.5 * dt * F[k]; R[k] += dt * V[k]; }
      acc = forces(R, F, g);
      for (let k = 0; k < 2 * NAT; k++) V[k] += 0.5 * dt * F[k];
    } else {
      for (let k = 0; k < 2 * NAT; k++) { R[k] += dt * V[k]; V[k] += dt * F[k]; }
      acc = forces(R, F, g);
    }
    sumLj += acc.lj; sumEx += acc.extra;
    const e = ke() + ljEnergy(R);
    last = e; maxDrift = Math.max(maxDrift, Math.abs(e - E0));
    let mx = 0, my = 0, size = 0;
    for (let i = 0; i < NAT; i++) { mx += R[2 * i] / NAT; my += R[2 * i + 1] / NAT; }
    for (let i = 0; i < NAT; i++) size = Math.max(size, Math.hypot(R[2 * i] - mx, R[2 * i + 1] - my));
    bLo = Math.min(bLo, e); bHi = Math.max(bHi, e);
    if (n % every === 0 || n === steps) save(e, n * dt);
    if (!Number.isFinite(e) || size > R_BLOW) { blown = true; if (n % every) save(e, n * dt); break; }
  }
  // 표류는 그림용 표본이 아니라 모든 스텝에서 잰다
  const ok = Number.isFinite(last);
  return { E0, E, lo, hi, t, frames, blown, steps: Math.min(n, steps), maxDrift: ok ? maxDrift : Infinity, endDrift: ok ? last - E0 : Infinity, forceErr: Math.sqrt(sumEx / Math.max(sumLj, 1e-300)) };
}

const sci = (v: number) => {
  if (!Number.isFinite(v)) return "∞";
  const a = Math.abs(v);
  const s = a >= 0.01 && a < 1000 ? a.toFixed(a < 1 ? 3 : 2) : a.toExponential(1);
  return (v < 0 ? "−" : "") + s.replace(/-/g, "−");
};

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "분자동역학 에너지 보존", "compare", narrow ? 1.25 : 0.5, "LJ 원자 일곱 개의 움직임과 총에너지 변화 E − E₀의 시계열");
  let integ: Integ = "verlet", dt = 0.005, nonCons = false;
  let sim = simulate(integ, dt, nonCons);
  let play = 0, playStart = 0, raf = 0;
  const DUR = 4000;

  function boxes() {
    const W = f.width, H = f.height, top = 24, pad = 10, gap = 18;
    if (W >= 460) {
      const side = Math.min(H - top - pad, W * 0.4);
      return { a: { x: pad, y: top, s: side }, b: { x: pad + side + gap + 44, y: top, w: W - 2 * pad - side - gap - 44, h: side - 18 } };
    }
    const side = Math.min(W - 2 * pad, (H - 2 * top - gap) * 0.5);
    const y2 = top + side + gap + top - 6;
    return { a: { x: (W - side) / 2, y: top, s: side }, b: { x: pad + 44, y: y2, w: W - 2 * pad - 44, h: H - y2 - 22 } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const { a, b } = boxes();
    const nF = sim.frames.length, fi = Math.min(nF - 1, Math.floor(play * (nF - 1)));
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.textBaseline = "alphabetic"; ctx.textAlign = "start";
    ctx.fillStyle = C.mid;
    ctx.fillText(`원자 배치 t = ${sim.t[fi].toFixed(1)}`, a.x, a.y - 8);
    ctx.fillText("총에너지 변화 E − E₀", b.x - 44, b.y - 8);

    // 원자: 무게중심을 따라가는 창(반폭 3), 지난 궤적은 꼬리로
    const Rf = sim.frames[fi];
    let cx = 0, cy = 0; for (let i = 0; i < NAT; i++) { cx += Rf[2 * i] / NAT; cy += Rf[2 * i + 1] / NAT; }
    const half = 3, sc = a.s / (2 * half);
    const P = (x: number, y: number): [number, number] => [a.x + a.s / 2 + (x - cx) * sc, a.y + a.s / 2 - (y - cy) * sc];
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(a.x, a.y, a.s, a.s);
    ctx.save(); ctx.beginPath(); ctx.rect(a.x, a.y, a.s, a.s); ctx.clip();
    const tail = Math.min(fi, 40);
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1.2;
    for (let i = 0; i < NAT; i++) {
      ctx.beginPath();
      for (let k = fi - tail; k <= fi; k++) { const q = P(sim.frames[k][2 * i], sim.frames[k][2 * i + 1]); k === fi - tail ? ctx.moveTo(...q) : ctx.lineTo(...q); }
      ctx.stroke();
    }
    const rad = Math.max(5, 0.5 * sc * 0.9);
    for (let i = 0; i < NAT; i++) {
      const [px, py] = P(Rf[2 * i], Rf[2 * i + 1]);
      ctx.fillStyle = sim.blown && fi === nF - 1 ? C.edge : C.ink2; ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, rad, 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.restore();

    // 에너지 시계열: 세로축은 지금 궤적의 max|E − E₀|에 맞춘다
    let amp = 0; for (const e of [...sim.lo, ...sim.hi]) if (Number.isFinite(e)) amp = Math.max(amp, Math.abs(e - sim.E0));
    amp = Math.max(amp, 1e-7) * 1.15;
    const X = (t: number) => b.x + (t / T_END) * b.w, Y = (v: number) => b.y + b.h / 2 - (Math.max(-amp, Math.min(amp, v)) / amp) * (b.h / 2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.beginPath(); ctx.moveTo(b.x, Y(0)); ctx.lineTo(b.x + b.w, Y(0)); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "right";
    const tick = amp / 1.15;
    [tick, 0, -tick].forEach((v) => ctx.fillText(v === 0 ? "0" : sci(v), b.x - 4, Y(v) + 3));
    ctx.textAlign = "center";
    [0, 4, 8, 12, 16].forEach((v) => ctx.fillText(String(v), X(v), b.y + b.h + 13));
    ctx.textAlign = "start";
    ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    sim.E.forEach((e, k) => { const q: [number, number] = [X(sim.t[k]), Y(e - sim.E0)]; k ? ctx.lineTo(...q) : ctx.moveTo(...q); });
    ctx.stroke();
    // 저장 구간의 최소·최대 봉투
    ctx.fillStyle = C.edge; ctx.beginPath();
    sim.hi.forEach((e, k) => { const q: [number, number] = [X(sim.t[k]), Y(e - sim.E0)]; k ? ctx.lineTo(...q) : ctx.moveTo(...q); });
    for (let k = sim.lo.length - 1; k >= 0; k--) ctx.lineTo(X(sim.t[k]), Y(sim.lo[k] - sim.E0));
    ctx.closePath(); ctx.fill();
    // 재생 위치
    const [tx, ty] = [X(sim.t[fi]), Y(sim.E[fi] - sim.E0)];
    ctx.fillStyle = C.ink; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(tx, ty, 4.5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.restore();
    if (sim.blown) {
      ctx.font = "600 12px JetBrains Mono, monospace"; ctx.fillStyle = C.edge;
      ctx.fillText(`t = ${sim.t[nF - 1].toFixed(1)}에서 터짐`, Math.min(X(sim.t[nF - 1]) + 6, b.x + b.w - 130), b.y + 16);
    }
  }

  function frameLoop(now: number) {
    play = Math.max(0, Math.min(1, (now - playStart) / DUR));
    draw();
    raf = play < 1 ? requestAnimationFrame(frameLoop) : 0;
  }
  function replay() {
    if (reducedMotion()) { play = 1; draw(); return; }
    playStart = performance.now(); play = 0;
    if (!raf) raf = requestAnimationFrame(frameLoop);
  }

  function update(animate = true) {
    sim = simulate(integ, dt, nonCons);
    ro.set("steps", `스텝 ${sim.steps}번`);
    ro.set("max", `표류 max|E − E₀| ${sci(sim.maxDrift)}`, sim.maxDrift > 0.01 ? "hot" : "ok");
    ro.set("end", `끝 E − E₀ ${sim.endDrift > 0 ? "+" : ""}${sci(sim.endDrift)}`);
    ro.set("ferr", nonCons ? `더한 힘 / 참 힘 RMS ${(100 * sim.forceErr).toFixed(1)}%` : "힘 = −∇U");
    ro.set("blow", sim.blown ? "계가 터짐" : "계 유지", sim.blown ? "hot" : "");
    if (animate) replay(); else draw();
  }

  const ib = presets<Integ>([{ key: "verlet", label: "velocity Verlet" }, { key: "euler", label: "explicit Euler" }], (k) => { integ = k; update(); }, integ);
  ib.row.classList.add("segmented");
  const pd = param({
    label: "Δt", min: -3, max: Math.log10(0.02), step: 0.01, value: Math.log10(dt),
    fmt: (v) => (10 ** v).toPrecision(2), aria: "적분 시간 간격 Δt(로그 눈금, LJ 시간 단위)", onInput: (v) => { dt = 10 ** v; update(); },
  });
  const grid = el("div", "param-grid");
  grid.append(pd.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" /> 비보존력 섞기 (γ = ${GAMMA})`;
  sw.querySelector("input")!.addEventListener("change", (e) => { nonCons = (e.target as HTMLInputElement).checked; update(); });
  const again = el("button", "preset");
  again.type = "button";
  again.textContent = "▶ 다시 재생";
  again.addEventListener("click", replay);
  tools.append(sw, again);
  const ro = readouts();
  f.panel.append(ib.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
