// 격자 세계 가치 반복 (data/widgets.toml: gridworld-vi).
// 20.1장의 미끄러지는 4×4 격자: 오른쪽 위 칸(목표) +1, 그 아래 칸(함정) −1, 둘 다 흡수 상태.
// 행동 넷(위·오른쪽·아래·왼쪽), 1 − p는 의도대로, p/2씩 좌우로 미끄러진다(장의 예: p = 0.2). 격자 밖 이동은 제자리.
// V_{k+1} = max_a [R + γ P V_k]를 V_0 = 0에서 k번 적용한 값과 그 V_k에 대한 탐욕 정책을 그린다.
// 오른쪽(좁은 화면에서는 아래) 그림은 ‖V_k − V*‖∞와 수축 상한 γ^k ‖V_0 − V*‖∞(로그 눈금)다.
import { frame, param, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

export const H = 4, W = 4, S = 16, A = 4, GOAL = 3, PIT = 7, KMAX = 200;
const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];
const ARROW = ["↑", "→", "↓", "←"];

function move(s: number, d: number) {
  const r = Math.floor(s / W), c = s % W, r2 = r + MOVES[d][0], c2 = c + MOVES[d][1];
  return r2 >= 0 && r2 < H && c2 >= 0 && c2 < W ? r2 * W + c2 : s;
}

/** P[s][a] = [다음 상태, 확률] 목록, R[s][a] = 기대 즉시 보상 */
export function buildMdp(slip: number) {
  const P: [number, number][][][] = [], R: number[][] = [];
  for (let s = 0; s < S; s++) {
    P.push([]); R.push([]);
    for (let a = 0; a < A; a++) {
      if (s === GOAL || s === PIT) { P[s].push([[s, 1]]); R[s].push(0); continue; }
      const out: [number, number][] = [];
      for (const [d, p] of [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]] as [number, number][]) {
        if (p > 0) out.push([move(s, d), p]);
      }
      P[s].push(out);
      R[s].push(out.reduce((t, [s2, p]) => t + p * (s2 === GOAL ? 1 : s2 === PIT ? -1 : 0), 0));
    }
  }
  return { P, R };
}

export function qOf(m: ReturnType<typeof buildMdp>, V: Float64Array, gamma: number) {
  const Q: number[][] = [];
  for (let s = 0; s < S; s++) Q.push(m.R[s].map((r, a) => m.P[s][a].reduce((t, [s2, p]) => t + gamma * p * V[s2], r)));
  return Q;
}

/** V_0 = 0부터 KMAX번(또는 수렴까지)의 가치 반복 기록과 V* */
export function iterate(slip: number, gamma: number) {
  const m = buildMdp(slip);
  const hist: Float64Array[] = [new Float64Array(S)];
  let V = hist[0], conv = -1;
  for (let k = 0; k < 20000; k++) {
    const Q = qOf(m, V, gamma);
    const Vn = new Float64Array(S);
    let gap = 0;
    for (let s = 0; s < S; s++) { Vn[s] = Math.max(...Q[s]); gap = Math.max(gap, Math.abs(Vn[s] - V[s])); }
    V = Vn;
    if (k < KMAX) hist.push(Vn);
    if (conv < 0 && gap < 1e-8) conv = k + 1;     // 장의 기준: 연속 차이 < 1e-8
    if (gap < 1e-13) break;
  }
  const Vstar = V;
  const greedy = (v: Float64Array) => qOf(m, v, gamma).map((q) => {
    const mx = Math.max(...q), mn = Math.min(...q);
    return mx - mn < 1e-12 ? -1 : q.indexOf(mx);   // 모든 행동이 같으면 정하지 않는다
  });
  const piStar = greedy(Vstar);
  const err = hist.map((v) => v.reduce((t, x, s) => Math.max(t, Math.abs(x - Vstar[s])), 0));
  const pis = hist.map(greedy);
  const same = (pi: number[]) => pi.every((a, s) => s === GOAL || s === PIT || a === piStar[s]);
  const firstSame = pis.findIndex((pi, k) => pis.slice(k).every(same));
  return { m, hist, Vstar, piStar, pis, err, conv, firstSame };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "격자 세계 가치 반복", "step", narrow ? 1.12 : 0.55, "4×4 격자의 칸마다 가치 V_k를 색과 숫자로, 탐욕 정책을 화살표로 그리고, 옆에 V*까지의 오차를 로그 눈금으로 그린 그림");
  let gamma = 0.95, slip = 0.2, k = 5;
  let res = iterate(slip, gamma);

  function draw() {
    const { ctx, width: Wd, height: Ht } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, Wd, Ht);
    const stack = Wd <= 400;
    const side = stack ? Math.min(Wd - 24, Ht * 0.58) : Math.min(Ht - 20, Wd * 0.56);
    const gx = stack ? (Wd - side) / 2 : 10, gy = 10, cell = side / W;
    const V = res.hist[Math.min(k, res.hist.length - 1)], pi = res.pis[Math.min(k, res.pis.length - 1)];
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let s = 0; s < S; s++) {
      const r = Math.floor(s / W), c = s % W, x = gx + c * cell, y = gy + r * cell;
      const v = s === GOAL ? 1 : s === PIT ? -1 : V[s];
      ctx.fillStyle = alpha(v >= 0 ? C.edge : C.mid, Math.min(1, Math.abs(v)) * (v >= 0 ? 0.8 : 0.6));
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      ctx.fillStyle = C.ink;
      if (s === GOAL || s === PIT) {
        ctx.font = `700 ${Math.round(cell * 0.3)}px Pretendard Variable, sans-serif`;
        ctx.fillText(s === GOAL ? "+1" : "−1", x + cell / 2, y + cell / 2);
        continue;
      }
      ctx.font = `${Math.round(Math.max(10, cell * 0.17))}px JetBrains Mono, monospace`;
      ctx.fillText((v + 0).toFixed(2).replace("-", "−"), x + cell / 2, y + cell * 0.76);
      const a = pi[s];
      ctx.font = `700 ${Math.round(cell * 0.34)}px Pretendard Variable, sans-serif`;
      if (a < 0) { ctx.fillStyle = C.mid; ctx.fillText("·", x + cell / 2, y + cell * 0.38); }
      else { ctx.fillStyle = a === res.piStar[s] ? C.ink : C.mid; ctx.fillText(ARROW[a], x + cell / 2, y + cell * 0.38); }
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    // 오차 그림: log10 ‖V_k − V*‖∞
    const px0 = stack ? 40 : gx + side + 44, px1 = Wd - 10;
    const py0 = stack ? gy + side + 26 : 22, py1 = Ht - 24;
    const L0 = -8, L1 = 0.5;
    const X = (kk: number) => px0 + (kk / KMAX) * (px1 - px0);
    const Y = (e: number) => py1 - ((Math.max(L0, Math.min(L1, Math.log10(Math.max(e, 1e-12)))) - L0) / (L1 - L0)) * (py1 - py0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const e of [0, -2, -4, -6, -8]) {
      ctx.beginPath(); ctx.moveTo(px0, Y(10 ** e)); ctx.lineTo(px1, Y(10 ** e)); ctx.stroke();
      ctx.fillText(e === 0 ? "1" : `1e${e}`.replace("-", "−"), px0 - 34, Y(10 ** e) + 4);
    }
    for (const kk of [0, 100, 200]) ctx.fillText(String(kk), Math.min(X(kk) - 4, px1 - 22), py1 + 15);
    ctx.fillText("‖V_k − V*‖∞", px0 + 4, py0 - 6);
    // 수축 상한 γ^k ‖V_0 − V*‖∞
    ctx.strokeStyle = C.mid; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let kk = 0; kk <= KMAX; kk += 2) { const y = Y(gamma ** kk * res.err[0]); if (kk) ctx.lineTo(X(kk), y); else ctx.moveTo(X(kk), y); }
    ctx.stroke(); ctx.setLineDash([]);
    const lb = "γ^k 상한";
    ctx.fillText(lb, px1 - ctx.measureText(lb).width, Math.max(py0 + 10, Math.min(py1 - 4, Y(gamma ** KMAX * res.err[0]) - 6)));
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath();
    res.err.forEach((e, kk) => { if (kk) ctx.lineTo(X(kk), Y(e)); else ctx.moveTo(X(kk), Y(e)); });
    ctx.stroke();
    const kk = Math.min(k, res.err.length - 1);
    ctx.fillStyle = C.edge;
    ctx.beginPath(); ctx.arc(X(kk), Y(res.err[kk]), 4.5, 0, 7); ctx.fill();
  }

  let raf = 0;
  function update(recompute: boolean) {
    if (recompute) res = iterate(slip, gamma);
    const kk = Math.min(k, res.hist.length - 1);
    const V = res.hist[kk];
    ro.set("err", `‖V_${k} − V*‖∞ = ${res.err[kk] < 1e-12 ? "0" : res.err[kk].toExponential(1).replace("-", "−")}`, res.err[kk] < 1e-8 ? "ok" : "");
    ro.set("conv", res.conv > 0 ? `수렴(차이 < 1e−8) ${res.conv}번` : "수렴 안 함");
    ro.set("pi", res.firstSame >= 0 ? `정책은 k = ${res.firstSame}부터 π*` : `k ≤ ${KMAX}에서 π*와 다름`, k >= res.firstSame && res.firstSame >= 0 ? "ok" : "hot");
    ro.set("v0", `V_k(3,0) = ${V[12].toFixed(3)} · V* = ${res.Vstar[12].toFixed(3)}`);
    pageState.set("gridworld", { gamma, slip, k, conv: res.conv }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pk = param({ label: "반복", min: 0, max: KMAX, step: 1, value: k, fmt: (v) => String(v), aria: "가치 반복 횟수 k", onInput: (v) => { k = v; update(false); } });
  const pg = param({ label: "γ", min: 0.5, max: 0.99, step: 0.01, value: gamma, fmt: (v) => v.toFixed(2), aria: "할인율 γ", onInput: (v) => { gamma = v; update(true); } });
  const ps = param({ label: "미끄럼", min: 0, max: 0.6, step: 0.05, value: slip, fmt: (v) => v.toFixed(2), aria: "좌우로 미끄러질 확률의 합 p", onInput: (v) => { slip = v; update(true); } });
  const grid = el("div", "param-grid");
  grid.append(pk.row, pg.row, ps.row);
  for (const r of [pk.row, pg.row, ps.row]) r.style.gridTemplateColumns = "3.4em minmax(0, 1fr) 3.2em";
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update(false);
}
