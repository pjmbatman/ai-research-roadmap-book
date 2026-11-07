// GNN over-smoothing (data/widgets.toml: over-smoothing).
// 18.6장 "한계"의 설정: 공동체 2개 × 20 노드(같은 공동체끼리 연결 확률 0.25, 다른 공동체 0.05), 특징 8차원,
// X = μ_y + 2.0·N(0, I), 학습 파라미터 없는 GCN 전파 H ← Â H, Â = D̃^{−1/2}(A + I)D̃^{−1/2}. 그래프와 잡음은 위젯의 고정 시드다.
// 잔차 연결을 켜면 입력으로 돌아가는 초기 잔차 H ← (1 − α)ÂH + αX(α = 0.2)를 쓴다.
// 노드 색은 방향만 남긴 표현 ĥ_i = h_i/‖h_i‖를 두 공동체 입력 평균의 차 방향에 투영한 값이다(모양 = 참 공동체).
// Dirichlet 에너지 ½Σ A_ij‖ĥ_i − ĥ_j‖²는 방향만 남긴 표현으로 잰다(Â 전파는 크기를 √d̃_i에 비례하게 남기므로).
import { frame, param, readouts, el, C, rng, alpha, rgbOf } from "../frame";

export const NN = 40, D = 8, KMAX = 64, ALPHA = 0.2;
export const G = (() => {
  const r = rng(2330);   // 정확도 변화가 본문 셀(0.88 → 1.00 → 0.62)에 가까운 시드
  const y = Array.from({ length: NN }, (_, i) => (i < NN / 2 ? 0 : 1));
  const A = Array.from({ length: NN }, () => new Float64Array(NN));
  for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) if (r.u() < (y[i] === y[j] ? 0.25 : 0.05)) A[i][j] = A[j][i] = 1;
  const mu = [0, 1].map(() => Array.from({ length: D }, () => r.normal()));
  const X = y.map((c) => mu[c].map((m) => m + 2.0 * r.normal()));
  const deg = A.map((row) => row.reduce((s, v) => s + v, 0) + 1);
  const An = A.map((row, i) => Float64Array.from(row, (v, j) => (v + (i === j ? 1 : 0)) / Math.sqrt(deg[i] * deg[j])));
  return { y, A, X, An, mu };
})();

const unit = (h: number[]) => { const n = Math.hypot(...h) || 1; return h.map((v) => v / n); };

/** 전파 0..KMAX번의 표현과 지표 */
export function propagate(residual: boolean) {
  const { An, X, y, A } = G;
  const Hs: number[][][] = [X.map((r) => r.slice())];
  for (let k = 1; k <= KMAX; k++) {
    const P = Hs[k - 1];
    Hs.push(An.map((row, i) => Array.from({ length: D }, (_, d) => {
      let s = 0; for (let j = 0; j < NN; j++) if (row[j]) s += row[j] * P[j][d];
      return residual ? (1 - ALPHA) * s + ALPHA * X[i][d] : s;
    })));
  }
  const stats = Hs.map((H) => {
    const c = [0, 1].map((k) => Array.from({ length: D }, (_, d) => H.filter((_, i) => y[i] === k).reduce((s, h) => s + h[d], 0) / (NN / 2)));
    const acc = H.filter((h, i) => {
      const d0 = h.reduce((s, v, d) => s + (v - c[0][d]) ** 2, 0), d1 = h.reduce((s, v, d) => s + (v - c[1][d]) ** 2, 0);
      return (d1 < d0 ? 1 : 0) === y[i];
    }).length / NN;
    const U = H.map(unit);
    let cos = 0, n = 0, E = 0;
    for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) {
      if (y[i] !== y[j]) { cos += U[i].reduce((s, v, d) => s + v * U[j][d], 0); n++; }
      if (A[i][j]) E += 0.5 * U[i].reduce((s, v, d) => s + (v - U[j][d]) ** 2, 0);
    }
    return { acc, cos: cos / n, E };
  });
  return { Hs, stats };
}

/** 고정 시드의 Fruchterman–Reingold 배치(공동체 0은 왼쪽, 1은 오른쪽에서 시작) */
function layout() {
  const r = rng(77);
  const P = G.y.map((c) => [(c ? 0.5 : -0.5) + 0.3 * r.normal(), 0.3 * r.normal()]);
  const k = 0.22;
  for (let it = 0; it < 300; it++) {
    const t = 0.08 * (1 - it / 300) + 0.002;
    const F = P.map(() => [0, 0]);
    for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) {
      if (i === j) continue;
      const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d = Math.hypot(dx, dy) + 1e-6;
      const rep = (k * k) / d, att = G.A[i][j] ? (d * d) / k : 0;
      F[i][0] += (dx / d) * (rep - att); F[i][1] += (dy / d) * (rep - att);
    }
    P.forEach((p, i) => { const m = Math.hypot(F[i][0], F[i][1]) || 1; p[0] += (F[i][0] / m) * Math.min(m, t); p[1] += (F[i][1] / m) * Math.min(m, t); p[0] *= 0.995; p[1] *= 0.995; });
  }
  return P;
}

export function mount(root: HTMLElement) {
  const f = frame(root, "GNN over-smoothing", "manipulate", 0.6, "두 공동체 그래프의 노드 색(특징 방향)과 전파 횟수에 따른 정확도·Dirichlet 에너지");
  let K = 0, residual = false;
  let run = propagate(false);
  const runs = { off: run, on: propagate(true) };
  const pos = layout();
  const u = unit(G.mu[1].map((v, d) => v - G.mu[0][d]));
  const proj = (h: number[]) => unit(h).reduce((s, v, d) => s + v * u[d], 0);
  const S0 = Math.max(...G.X.map((h) => Math.abs(proj(h))));

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const split = Math.round(H * 0.68);
    // 그래프
    const xs = pos.map((p) => p[0]), ys = pos.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sc = Math.min((W - 40) / (x1 - x0 || 1), (split - 30) / (y1 - y0 || 1));
    const ox = W / 2 - ((x0 + x1) / 2) * sc, oy = (split + 4) / 2 - ((y0 + y1) / 2) * sc;
    const P = (i: number) => [ox + pos[i][0] * sc, oy + pos[i][1] * sc];
    ctx.strokeStyle = alpha(C.mid, 0.28); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < NN; i++) for (let j = i + 1; j < NN; j++) if (G.A[i][j]) { const [a, b] = P(i), [c, d] = P(j); ctx.moveTo(a, b); ctx.lineTo(c, d); }
    ctx.stroke();
    const Hk = run.Hs[K];
    const neg = rgbOf(C.ink2), posc = rgbOf(C.edge), mid = rgbOf(C.faint);
    const col = (t: number) => { const e = t < 0 ? neg : posc, a = Math.min(1, Math.abs(t)); return `rgb(${mid.map((m, i) => Math.round(m + (e[i] - m) * a)).join(",")})`; };
    const rad = Math.max(5, Math.min(9, W / 70));
    for (let i = 0; i < NN; i++) {
      const [x, y] = P(i);
      ctx.fillStyle = col(proj(Hk[i]) / S0);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
      ctx.beginPath();
      if (G.y[i] === 0) ctx.arc(x, y, rad, 0, 7); else ctx.rect(x - rad * 0.9, y - rad * 0.9, rad * 1.8, rad * 1.8);
      ctx.fill(); ctx.stroke();
    }
    // 아래: 전파 횟수(로그 눈금)에 따른 정확도와 E/E₀
    const L = 34, R = W - 12, top = split + 20, bot = H - 18;
    const X = (k: number) => L + (Math.log2(1 + k) / Math.log2(1 + KMAX)) * (R - L);
    const Y = (v: number) => bot - v * (bot - top);
    ctx.strokeStyle = C.faint;
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(R, Y(0)); ctx.moveTo(L, Y(1)); ctx.lineTo(R, Y(1)); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.fillText("0", L - 12, Y(0) + 4); ctx.fillText("1", L - 12, Y(1) + 4);
    for (const k of [0, 1, 4, 16, 64]) { const t = String(k), tw = ctx.measureText(t).width; ctx.fillText(t, X(k) - tw / 2, bot + 13); }
    const E0 = run.stats[0].E;
    const curve = (fn: (s: (typeof run.stats)[number]) => number, c: string) => {
      ctx.strokeStyle = c; ctx.lineWidth = 1.8;
      ctx.beginPath(); run.stats.forEach((s, k) => (k ? ctx.lineTo(X(k), Y(fn(s))) : ctx.moveTo(X(k), Y(fn(s))))); ctx.stroke();
    };
    curve((s) => s.acc, C.ink2);
    curve((s) => Math.min(1, s.E / E0), C.edge);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(K), top - 4); ctx.lineTo(X(K), bot); ctx.stroke(); ctx.setLineDash([]);
    const l2 = "E/E₀", w2 = ctx.measureText(l2).width, l1 = "정확도", w1 = ctx.measureText(l1).width;
    ctx.fillStyle = C.edge; ctx.fillText(l2, R - w2, top - 6);
    ctx.fillStyle = C.ink2; ctx.fillText(l1, R - w2 - w1 - 12, top - 6);
  }

  let raf = 0;
  function update() {
    run = residual ? runs.on : runs.off;
    const s = run.stats[K], E0 = run.stats[0].E;
    ro.set("k", `전파 ${K}번`);
    ro.set("acc", `분류 정확도 ${s.acc.toFixed(2)}`, s.acc >= 0.95 ? "ok" : "");
    ro.set("cos", `다른 공동체 간 cos ${s.cos >= 0 ? "+" : ""}${s.cos.toFixed(3)}`, s.cos > 0.95 ? "hot" : "");
    ro.set("E", `Dirichlet E/E₀ ${(s.E / E0).toFixed(3)}`, s.E / E0 < 0.01 ? "hot" : "");
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pk = param({ label: "층", min: 0, max: KMAX, step: 1, value: K, fmt: (v) => `${v}`, aria: "전파 층 수", onInput: (v) => { K = v; update(); } });
  pk.row.style.gridTemplateColumns = "2.4em minmax(0, 1fr) 2.6em";
  const grid = el("div", "param-grid");
  grid.append(pk.row);
  const tools = el("div", "wframe-row");
  const tg = el("label", "toggle");
  tg.innerHTML = `<input type="checkbox" /> 잔차 연결 (1 − α)ÂH + αX, α = ${ALPHA}`;
  tg.querySelector("input")!.addEventListener("change", (e) => { residual = (e.target as HTMLInputElement).checked; update(); });
  tools.append(tg);
  const ro = readouts();
  f.panel.append(grid, tools, ro.row);
  f.onResize(draw);
  update();
}
