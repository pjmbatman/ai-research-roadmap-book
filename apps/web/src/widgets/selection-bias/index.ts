// 시도 횟수와 test 낙관 편향 (data/widgets.toml: selection-bias).
// 4.6장의 시뮬레이션과 같은 설정: 모든 설정의 실제 정확도는 똑같이 p = 0.70이고, test 표본 n개로 정확도를 잰다.
// 설정 k개의 test 정확도는 서로 독립인 Binomial(n, p)/n이다. test 결과를 보고 최고를 고르면 그 값의 분포는
// 최댓값의 분포 P(max ≤ x) = F(x)^k를 따른다(F는 이항분포의 누적분포). 고른 설정을 새 데이터로 다시 재면 다시 Binomial(n, p)/n이다.
// 기대값은 근사 없이 이항분포의 확률질량으로 정확히 계산한다. 아래 줄의 점은 시드를 고정한 한 번의 추첨이다.
import { frame, param, readouts, C, alpha, rng } from "../frame";

export const P_TRUE = 0.7;
export const KS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200];
export const NS = [50, 100, 200, 500, 1000, 2000];

/** Binomial(n, p)의 확률질량과 누적분포(로그 공간에서 계산해 n = 2000에서도 넘치지 않는다) */
export function binom(n: number, p: number) {
  const lf = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) lf[i] = lf[i - 1] + Math.log(i);
  const pmf = new Float64Array(n + 1), cdf = new Float64Array(n + 1);
  let c = 0;
  for (let x = 0; x <= n; x++) {
    pmf[x] = Math.exp(lf[n] - lf[x] - lf[n - x] + x * Math.log(p) + (n - x) * Math.log(1 - p));
    c += pmf[x];
    cdf[x] = Math.min(c, 1);
  }
  return { pmf, cdf };
}
/** k개 중 최고 test 정확도의 확률질량과 기대값 */
export function maxOf(n: number, k: number, b = binom(n, P_TRUE)) {
  const q = new Float64Array(n + 1);
  let e = 0;
  for (let x = 0; x <= n; x++) {
    q[x] = Math.pow(b.cdf[x], k) - (x ? Math.pow(b.cdf[x - 1], k) : 0);
    e += (x / n) * q[x];
  }
  return { q, mean: e };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 560;
  const f = frame(root, "시도 횟수와 test 낙관 편향", "manipulate", narrow ? 0.95 : 0.5, "설정 하나의 test 정확도 분포와, k개 중 최고로 고른 설정의 test 정확도 분포, 그리고 한 번의 추첨에서 고른 설정이 새 데이터에서 받은 점수");
  let ki = KS.indexOf(20), ni = NS.indexOf(200);
  // 고정한 난수열: 설정 i의 test 점수와 새 데이터 점수에 쓸 균등난수(k를 늘리면 점이 더해질 뿐 앞의 점은 그대로다)
  const r = rng(4060);
  const U = Array.from({ length: 200 }, () => r.u()), UF = r.u(), JIT = Array.from({ length: 200 }, () => r.u());
  let cache: { n: number; b: ReturnType<typeof binom> } | null = null;
  const B = (n: number) => (cache && cache.n === n ? cache.b : (cache = { n, b: binom(n, P_TRUE) }).b);
  const quantile = (cdf: Float64Array, u: number) => { let x = 0; while (x < cdf.length - 1 && cdf[x] < u) x++; return x; };

  function sample() {
    const n = NS[ni], k = KS[ki], b = B(n);
    const accs = U.slice(0, k).map((u) => quantile(b.cdf, u) / n);
    let best = 0; accs.forEach((a, i) => { if (a > accs[best]) best = i; });
    return { accs, best, fresh: quantile(b.cdf, UF) / n };
  }

  const X0 = 0.5, X1 = 0.95;
  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const n = NS[ni], k = KS[ki], b = B(n), m = maxOf(n, k, b), s = sample();
    const padL = 16, padR = 16, top = 48, stripH = Math.min(70, H * 0.22), axisY = H - stripH - 30;
    const X = (a: number) => padL + ((a - X0) / (X1 - X0)) * (W - padL - padR);
    // 밀도(정확도 단위당) = 확률질량 × n
    let ymax = 0;
    for (let x = 0; x <= n; x++) ymax = Math.max(ymax, b.pmf[x] * n, m.q[x] * n);
    const Y = (d: number) => axisY - (d / (ymax * 1.08)) * (axisY - top);
    const curve = (arr: Float64Array, fill: string, stroke: string) => {
      ctx.beginPath();
      ctx.moveTo(X(X0), Y(0));
      for (let x = 0; x <= n; x++) {
        const a = x / n;
        if (a < X0 - 1 / n || a > X1 + 1 / n) continue;
        const l = Math.max(X0, a - 0.5 / n), rr = Math.min(X1, a + 0.5 / n);
        ctx.lineTo(X(l), Y(arr[x] * n)); ctx.lineTo(X(rr), Y(arr[x] * n));
      }
      ctx.lineTo(X(X1), Y(0));
      ctx.closePath();
      ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = 1.6; ctx.stroke();
    };
    // 축
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, axisY); ctx.lineTo(W - padR, axisY); ctx.stroke();
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "center";
    const tick = W < 460 ? 0.1 : 0.05;
    for (let a = 0.5; a <= 0.951; a += tick) { ctx.fillText(a.toFixed(2), X(a), axisY + 13); ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(X(a), axisY); ctx.lineTo(X(a), axisY + 3); ctx.stroke(); }
    ctx.textAlign = "start";
    // 분포 둘: 설정 하나(= 새 데이터에서의 점수) · k개 중 최고
    curve(b.pmf, alpha(C.ink2, 0.22), C.ink2);
    if (k > 1) curve(m.q, alpha(C.edge, 0.25), C.edge);
    // 참값과 기대 최고값
    ctx.setLineDash([4, 4]); ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X(P_TRUE), top - 16); ctx.lineTo(X(P_TRUE), axisY); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(X(m.mean), top - 16); ctx.lineTo(X(m.mean), axisY); ctx.stroke();
    ctx.font = "600 11px Pretendard Variable, sans-serif"; ctx.textBaseline = "middle";
    ctx.fillStyle = C.mid; ctx.textAlign = "right"; ctx.fillText(`참값 ${P_TRUE.toFixed(2)}`, X(P_TRUE) - 5, top - 10);
    ctx.fillStyle = C.edge; ctx.textAlign = "left";
    if (k > 1) ctx.fillText(`최고의 기대값 ${m.mean.toFixed(3)}`, Math.min(X(m.mean) + 5, W - padR - 120), top - 10);
    ctx.textAlign = "start";
    // 범례
    ctx.font = "600 11px Pretendard Variable, sans-serif";
    const lg = (x: number, col: string, t: string) => { ctx.fillStyle = col; ctx.fillRect(x, 10, 10, 10); ctx.fillStyle = C.ink; ctx.fillText(t, x + 14, 15); return x + 14 + ctx.measureText(t).width + 14; };
    let lx = padL;
    lx = lg(lx, C.ink2, narrow ? "설정 하나" : "설정 하나 · 새 데이터");
    if (k > 1) lg(lx, C.edge, `${k}개 중 최고`);
    // 한 번의 추첨: 설정 k개의 test 점수와, 고른 설정의 새 데이터 점수
    const sy0 = axisY + 26, sy1 = H - 10;
    ctx.font = "600 11px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ctx.fillText("한 번의 추첨", padL, sy0 + 2);
    const rowY = (j: number) => sy0 + 14 + j * (sy1 - sy0 - 14);
    s.accs.forEach((a, i) => {
      if (i === s.best) return;
      ctx.fillStyle = alpha(C.ink2, 0.55);
      ctx.beginPath(); ctx.arc(X(Math.max(X0, Math.min(X1, a))), rowY(JIT[i] * 0.55), 3, 0, 7); ctx.fill();
    });
    const bx = X(Math.min(X1, s.accs[s.best])), fx = X(Math.max(X0, Math.min(X1, s.fresh))), by = rowY(0.25), fy = rowY(0.85);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(fx, fy); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(bx, by, 5.5, 0, 7); ctx.fill();
    ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(fx, fy, 5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.textBaseline = "alphabetic";
  }

  const pk = param({ label: "k", min: 0, max: KS.length - 1, step: 1, value: ki, fmt: (v) => `${KS[v]}개`, aria: "test로 비교한 설정 수 k", onInput: (v) => { ki = v; update(); } });
  const pn = param({ label: "n", min: 0, max: NS.length - 1, step: 1, value: ni, fmt: (v) => `${NS[v]}`, aria: "test 표본 수 n(클수록 평가 잡음이 작다)", onInput: (v) => { ni = v; update(); } });
  const grid = document.createElement("div");
  grid.className = "param-grid";
  grid.append(pk.row, pn.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);

  function update() {
    const n = NS[ni], k = KS[ki], m = maxOf(n, k, B(n)), s = sample();
    const sd = Math.sqrt((P_TRUE * (1 - P_TRUE)) / n);
    ro.set("noise", `설정 ${k}개 · test ${n}개 · 평가 잡음 σ ${sd.toFixed(3)}`);
    ro.set("best", `고른 설정의 기대 test 정확도 ${m.mean.toFixed(3)}`, "hot");
    ro.set("fresh", `새 데이터에서 기대값 ${P_TRUE.toFixed(3)}`, "ok");
    ro.set("bias", `낙관 편향 +${((m.mean - P_TRUE) * 100).toFixed(1)}%p`, m.mean - P_TRUE > 0.005 ? "hot" : "");
    ro.set("draw", `이번 추첨: test ${s.accs[s.best].toFixed(3)} → 새 데이터 ${s.fresh.toFixed(3)}`);
    draw();
  }
  f.onResize(draw);
  update();
}
