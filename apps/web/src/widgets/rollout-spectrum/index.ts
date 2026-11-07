// rollout 오차와 스펙트럼 (data/widgets.toml: rollout-spectrum).
// 26.5장과 같은 두 학습 시뮬레이터(A 직접 갱신 21개 특징, B 보존형 유속 3개 특징)를 같은 학습 궤적 36개로 맞추고,
// 시험 궤적(시드 100–105, ν = 0.07)에서 t = 1(50 step)까지 굴린다.
// 위: 고른 step의 장 u(x). 아래 왼쪽: 시험 궤적 6개의 rollout 상대 오차(로그). 아래 오른쪽: 에너지 스펙트럼 |û_k|²(로그), k ≥ 8은 고파수.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { N, T, DT, dx, fit, traj, rollout, rel, spectrum, hiK, type Model } from "./model";

type Key = "A" | "B";
const NU = 0.07;

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "rollout 오차와 스펙트럼", "manipulate", narrow ? 1.15 : 0.62, "학습된 시뮬레이터의 rollout 장, 시험 궤적별 오차 성장, 기준 해와 비교한 에너지 스펙트럼");
  let key: Key = "A", n = T, tr = 3; // 시드 103: A가 발산하지 않고 고파수 에너지를 쌓는 궤적
  const M = fit();
  const refs = Array.from({ length: 6 }, (_, i) => traj(12 + i, NU));
  const models: Record<Key, Model> = { A: M.A, B: M.B };
  const runs = {} as Record<Key, { states: Float64Array[]; blown: number; err: number[]; oneStep: number }[]>;
  for (const k of ["A", "B"] as Key[]) {
    runs[k] = refs.map((ref) => {
      const r = rollout(models[k], ref[0], NU);
      let e1 = 0; for (let i = 0; i < T; i++) e1 += rel(models[k](ref[i], NU), ref[i + 1]);
      return { ...r, err: r.states.map((u, i) => rel(u, ref[i])), oneStep: e1 / T };
    });
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const R = runs[key][tr], ref = refs[tr][n];
    const alive = n < R.states.length;
    const u = alive ? R.states[n] : null;
    // ── 위: 장
    const PL = 30, PR = 10, t0 = 26, t1 = Math.round(H * 0.44);
    let lo = Infinity, hi = -Infinity;
    for (const v of refs[tr][0]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 0.35 * (hi - lo); lo -= pad; hi += pad;
    const X = (j: number) => PL + (j / N) * (W - PL - PR), Y = (v: number) => t1 - ((Math.min(Math.max(v, lo - 1), hi + 1) - lo) / (hi - lo)) * (t1 - t0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(PL, t0, W - PL - PR, t1 - t0);
    ctx.fillStyle = C.mid;
    for (const v of [Math.ceil(lo * 2) / 2, Math.floor(hi * 2) / 2]) { const s = v.toFixed(1); ctx.fillText(s, PL - 4 - ctx.measureText(s).width, Y(v) + 4); }
    ctx.save(); ctx.beginPath(); ctx.rect(PL, t0, W - PL - PR, t1 - t0); ctx.clip();
    const line = (v: Float64Array, col: string, w: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); for (let j = 0; j <= N; j++) { const y = Y(v[j % N]); j ? ctx.lineTo(X(j), y) : ctx.moveTo(X(j), y); } ctx.stroke(); ctx.setLineDash([]);
    };
    line(ref, C.ink, 1.6, [5, 4]);
    if (u) line(u, C.edge, 2.2);
    ctx.restore();
    if (!u) { ctx.fillStyle = C.edge; ctx.font = "600 13px Pretendard Variable, sans-serif"; const s = `step ${R.blown}에서 발산`; ctx.fillText(s, (W - ctx.measureText(s).width) / 2, (t0 + t1) / 2); ctx.font = "11px JetBrains Mono, monospace"; }
    let lx = PL;
    for (const [nm, col, dash] of [["기준 solver", C.ink, true], [key === "A" ? "A 직접 갱신" : "B 보존형 유속", C.edge, false]] as [string, string, boolean][]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, 12); ctx.lineTo(lx + 14, 12); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, 16);
      lx += 30 + ctx.measureText(nm).width;
    }
    const tt = `t = ${(n * DT).toFixed(2)}`; ctx.fillStyle = C.mid; ctx.fillText(tt, W - PR - ctx.measureText(tt).width, 16);
    // ── 아래 왼쪽: rollout 상대 오차(로그) 대 step
    const b0 = t1 + 30, b1 = H - 22, split = Math.round(W * 0.52);
    const EX = (i: number) => PL + (i / T) * (split - PL - 6), EY = (e: number) => b1 - ((Math.log10(Math.min(Math.max(e, 1e-3), 10)) + 3) / 4) * (b1 - b0);
    ctx.strokeStyle = C.faint; ctx.strokeRect(PL, b0, split - PL - 6, b1 - b0);
    ctx.fillStyle = C.mid;
    for (const [e, s] of [[1e-3, "0.001"], [1e-1, "0.1"], [10, "10"]] as [number, string][]) {
      ctx.strokeStyle = alpha(C.mid, 0.15); ctx.beginPath(); ctx.moveTo(PL, EY(e)); ctx.lineTo(split - 6, EY(e)); ctx.stroke();
      ctx.fillText(s, PL + 3, EY(e) + (e === 10 ? 11 : -3));
    }
    ctx.fillText("rollout 오차", PL, b0 - 7);
    for (const s of [0, 25, 50]) { const t = String(s); ctx.fillText(t, Math.min(EX(s) - ctx.measureText(t).width / 2, split - 6 - ctx.measureText(t).width), b1 + 14); }
    runs[key].forEach((r, i) => {
      if (i === tr) return;
      ctx.strokeStyle = alpha(C.ink2, 0.6); ctx.lineWidth = 1.2;
      ctx.beginPath(); r.err.forEach((e, s) => { if (s === 0) return; s === 1 ? ctx.moveTo(EX(s), EY(e)) : ctx.lineTo(EX(s), EY(e)); }); ctx.stroke();
      if (r.blown) { const x = EX(r.blown), y = EY(10); ctx.beginPath(); ctx.moveTo(x - 3, y - 3); ctx.lineTo(x + 3, y + 3); ctx.moveTo(x + 3, y - 3); ctx.lineTo(x - 3, y + 3); ctx.stroke(); }
    });
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.4;
    ctx.beginPath(); R.err.forEach((e, s) => { if (s === 0) return; s === 1 ? ctx.moveTo(EX(s), EY(e)) : ctx.lineTo(EX(s), EY(e)); }); ctx.stroke();
    if (R.blown) { const x = EX(R.blown), y = EY(10); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke(); }
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(EX(n), b0); ctx.lineTo(EX(n), b1); ctx.stroke(); ctx.setLineDash([]);
    // ── 아래 오른쪽: 에너지 스펙트럼(로그), k = 1..32
    const sx0 = split + 30, sx1 = W - PR;
    const Er = spectrum(ref), Em = u ? spectrum(u) : null;
    const top = Math.log10(Math.max(...Er.slice(1))) + 1, bot = top - 11;
    const KX = (k: number) => sx0 + ((k - 1) / (N / 2 - 1)) * (sx1 - sx0), KY = (E: number) => b1 - ((Math.min(Math.max(Math.log10(Math.max(E, 1e-300)), bot), top) - bot) / (top - bot)) * (b1 - b0);
    ctx.fillStyle = C.fill; ctx.fillRect(KX(8), b0, sx1 - KX(8), b1 - b0);
    ctx.strokeStyle = C.faint; ctx.strokeRect(sx0, b0, sx1 - sx0, b1 - b0);
    ctx.fillStyle = C.mid;
    ctx.fillText("|û_k|²", sx0, b0 - 7);
    for (const k of [1, 8, 32]) { const s = String(k); ctx.fillText(s, Math.min(KX(k) - ctx.measureText(s).width / 2, sx1 - ctx.measureText(s).width), b1 + 14); }
    const spec = (E: number[], col: string, w: number, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
      ctx.beginPath(); for (let k = 1; k <= N / 2; k++) (k === 1 ? ctx.moveTo(KX(k), KY(E[k])) : ctx.lineTo(KX(k), KY(E[k]))); ctx.stroke(); ctx.setLineDash([]);
    };
    spec(Er, C.ink, 1.6, [5, 4]);
    if (Em) spec(Em, C.edge, 2.2);
  }

  let raf = 0;
  function update() {
    const R = runs[key][tr], ref = refs[tr];
    const alive = n < R.states.length;
    if (alive && n > 0) {
      const u = R.states[n];
      const hk = hiK(spectrum(u)) / hiK(spectrum(ref[n]));
      ro.set("now", `궤적 ${tr + 1} · 장 오차 ${R.err[n].toFixed(3)} · 고파수 비 ${hk.toFixed(2)}`, R.err[n] > 0.1 ? "hot" : "");
      const dm = Math.abs(u.reduce((a, b) => a + b, 0) - ref[0].reduce((a, b) => a + b, 0)) * dx;
      ro.set("mass", `질량 변화 ${dm < 1e-12 ? "< 1e-12" : dm.toExponential(1)}`, dm < 1e-12 ? "ok" : "");
    } else if (alive) {
      ro.set("now", `궤적 ${tr + 1} · 초기 상태`);
      ro.set("mass", "질량 변화 0");
    } else {
      ro.set("now", `궤적 ${tr + 1} · step ${R.blown}에서 발산`, "hot");
      ro.set("mass", "질량 변화 —");
    }
    const all = runs[key], blown = all.filter((r) => r.blown).length;
    ro.set("sum", `시험 6개: 한 step ${(all.reduce((s, r) => s + r.oneStep, 0) / 6).toFixed(4)} · 발산 ${blown}/6`);
    pageState.set("rollout-spectrum", { model: key, step: n, trajectory: tr + 100 }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pm = presets<Key>([{ key: "A", label: "A 직접 갱신(21)" }, { key: "B", label: "B 보존형 유속(3)" }], (k) => { key = k; update(); }, key);
  pm.row.classList.add("segmented");
  const ps = param({ label: "step", min: 0, max: T, step: 1, value: n, fmt: (v) => `${v}`, aria: "rollout step (Δt = 0.02, 50 step = t 1)", onInput: (v) => { n = v; update(); } });
  const pt = param({ label: "궤적", min: 1, max: 6, step: 1, value: tr + 1, fmt: (v) => `${v}`, aria: "시험 궤적 (시드 100–105)", onInput: (v) => { tr = v - 1; update(); } });
  const grid = el("div", "param-grid");
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  grid.append(ps.row, pt.row);
  const ro = readouts();
  f.panel.append(pm.row, grid, ro.row);
  f.onResize(draw);
  update();
}
