// 근접 중복 누수 임계 탐색 (data/widgets.toml: leakage-threshold).
// 합성 분자 지문(64비트, 비트 밀도 0.2)을 부모 400개에서 만들고, 부모마다 0~3개의 친척(비트 0~3개를 뒤집은 근접·정확 중복)을 붙인다.
// 물성은 y = 0.5 x·w + Σ c_ij x_i x_j(무작위 비트 쌍 60개) + 0.3 ε다. 무작위로 25%를 test로 나눈다.
// 모델 둘: 1-NN 암기 baseline(가장 가까운 train 분자의 레이블, 최근접 동점은 평균)과 선형 ridge(λ = 1, 절편은 벌점 없음).
// 임계값 t를 내리면 최근접 train Tanimoto 유사도가 t 이상인 test 분자를 뺀다. 남은 test에서 두 모델의 MAE를 잰다.
// 누수가 있으면 암기 baseline이 진짜 모델보다 좋아 보이고, 근접 중복을 빼면 순서가 뒤집힌다.
import { frame, param, readouts, el, C, rng, alpha } from "../frame";

export const NBIT = 64, PARENTS = 400, TMIN = 0.6, OFF = 1.1;
const popc = (v: number) => { v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; };

export function makeData() {
  const r = rng(2805);
  const fp: [number, number][] = [];
  const bitsOf = () => { const a: [number, number] = [0, 0]; for (let b = 0; b < NBIT; b++) if (r.u() < 0.2) a[b >> 5] |= 1 << (b & 31); return a; };
  for (let p = 0; p < PARENTS; p++) {
    const base = bitsOf();
    fp.push(base);
    const u = r.u(), m = u < 0.35 ? 0 : u < 0.65 ? 1 : u < 0.85 ? 2 : 3;
    for (let c = 0; c < m; c++) {
      const v: [number, number] = [base[0], base[1]];
      const k = Math.floor(r.u() * 4);
      const flipped = new Set<number>();
      while (flipped.size < k) flipped.add(Math.floor(r.u() * NBIT));
      flipped.forEach((b) => (v[b >> 5] ^= 1 << (b & 31)));
      fp.push(v);
    }
  }
  const N = fp.length;
  const bit = (n: number, b: number) => (fp[n][b >> 5] >>> (b & 31)) & 1;
  const w = Array.from({ length: NBIT }, () => r.normal());
  const pairs = Array.from({ length: 60 }, () => [Math.floor(r.u() * NBIT), Math.floor(r.u() * NBIT), 1.2 * r.normal()]);
  const y = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    let s = 0;
    for (let b = 0; b < NBIT; b++) s += 0.5 * w[b] * bit(n, b);
    for (const [i, j, c] of pairs) s += c * bit(n, i) * bit(n, j);
    y[n] = s + 0.3 * r.normal();
  }
  // 무작위 분할: test 25%
  const perm = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const te = perm.slice(0, Math.floor(N / 4)), tr = perm.slice(Math.floor(N / 4));
  return { fp, y, te, tr, N, bit };
}
export type Data = ReturnType<typeof makeData>;

export function tanimoto(a: [number, number], b: [number, number]) {
  const i = popc(a[0] & b[0]) + popc(a[1] & b[1]), u = popc(a[0] | b[0]) + popc(a[1] | b[1]);
  return u ? i / u : 1;
}

/** test 분자마다 최근접 train 유사도, 1-NN 예측, ridge 예측 */
export function evaluate(d: Data) {
  const { te, tr, fp, y, bit } = d;
  const sim = new Float64Array(te.length), nn = new Float64Array(te.length), lin = new Float64Array(te.length);
  te.forEach((t, k) => {
    let mx = -1, s = 0, c = 0;
    for (const n of tr) {
      const v = tanimoto(fp[t], fp[n]);
      if (v > mx + 1e-12) { mx = v; s = y[n]; c = 1; } else if (Math.abs(v - mx) <= 1e-12) { s += y[n]; c++; }
    }
    sim[k] = mx; nn[k] = s / c;
  });
  // ridge: (AᵀA + λD) β = Aᵀy, A = [X, 1], D = diag(1, …, 1, 0)
  const P = NBIT + 1, M = new Float64Array(P * P), rhs = new Float64Array(P), row = new Float64Array(P);
  for (const n of tr) {
    for (let b = 0; b < NBIT; b++) row[b] = bit(n, b);
    row[NBIT] = 1;
    for (let i = 0; i < P; i++) { if (!row[i]) continue; rhs[i] += y[n]; for (let j = 0; j < P; j++) M[i * P + j] += row[j]; }
  }
  for (let i = 0; i < NBIT; i++) M[i * P + i] += 1;
  const beta = solve(M, rhs, P);
  te.forEach((t, k) => { let s = beta[NBIT]; for (let b = 0; b < NBIT; b++) s += beta[b] * bit(t, b); lin[k] = s; });
  return { sim, nn, lin };
}
/** 가우스 소거(부분 피벗) */
function solve(M: Float64Array, b: Float64Array, n: number) {
  const A = M.slice(), x = b.slice();
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
    if (p !== c) { for (let j = 0; j < n; j++) [A[c * n + j], A[p * n + j]] = [A[p * n + j], A[c * n + j]]; [x[c], x[p]] = [x[p], x[c]]; }
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / A[c * n + c];
      if (!f) continue;
      for (let j = c; j < n; j++) A[r * n + j] -= f * A[c * n + j];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) { let s = x[r]; for (let j = r + 1; j < n; j++) s -= A[r * n + j] * x[j]; x[r] = s / A[r * n + r]; }
  return x;
}

/** 임계값 t에서 남은 test(최근접 유사도 < t)의 수와 두 모델의 MAE. t > 1이면 아무것도 빼지 않는다. */
export function atThreshold(d: Data, ev: ReturnType<typeof evaluate>, t: number) {
  let n = 0, a = 0, b = 0;
  d.te.forEach((i, k) => { if (t <= 1 && ev.sim[k] >= t) return; n++; a += Math.abs(ev.nn[k] - d.y[i]); b += Math.abs(ev.lin[k] - d.y[i]); });
  return { n, nn: n ? a / n : NaN, lin: n ? b / n : NaN };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 460;
  const f = frame(root, "근접 중복 누수 임계 탐색", "compare", narrow ? 1.3 : 0.52, "test 분자의 최근접 train 유사도 히스토그램과 임계값, 그리고 임계값에 따른 1-NN 암기 baseline과 선형 모델의 MAE 곡선");
  const d = makeData(), ev = evaluate(d);
  const T_GRID = Array.from({ length: 21 }, (_, i) => TMIN + (i * (1 - TMIN)) / 20);
  T_GRID.push(OFF);
  const curve = T_GRID.map((t) => atThreshold(d, ev, t));
  const full = atThreshold(d, ev, OFF);
  let t = OFF;

  function boxes() {
    const W = f.width, H = f.height, top = 24, pad = 10, gap = 22, lab = 34;
    if (W >= 460) {
      const w1 = (W - 2 * pad - gap) / 2, h = H - top - 26;
      return { a: { x: pad + lab, y: top, w: w1 - lab, h }, b: { x: pad + w1 + gap + lab, y: top, w: W - 2 * pad - w1 - gap - lab, h } };
    }
    const h = (H - 2 * top - gap - 30) / 2;
    return { a: { x: pad + lab, y: top, w: W - 2 * pad - lab, h }, b: { x: pad + lab, y: top * 2 + h + gap + 8, w: W - 2 * pad - lab, h } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const { a, b } = boxes();
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.textBaseline = "alphabetic"; ctx.textAlign = "start";
    ctx.fillStyle = C.mid;
    ctx.fillText("test 분자의 최근접 train 유사도", a.x - 34, a.y - 8);
    ctx.fillText("임계값에 따른 test MAE", b.x - 34, b.y - 8);

    // 히스토그램(폭 0.05): 남는 막대는 진하게, 빠지는 막대는 흐리게
    const NB = 20, cnt = new Array(NB).fill(0), cut = new Array(NB).fill(0);
    ev.sim.forEach((s) => { const k = Math.min(NB - 1, Math.floor(s * NB)); cnt[k]++; if (t <= 1 && s >= t) cut[k]++; });
    const cmax = Math.max(...cnt);
    const Xa = (v: number) => a.x + v * a.w, Ya = (c: number) => a.y + a.h - (c / cmax) * a.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(a.x, a.y, a.w, a.h);
    for (let k = 0; k < NB; k++) {
      const x0 = Xa(k / NB) + 1, bw = a.w / NB - 2;
      ctx.fillStyle = alpha(C.edge, 0.18); ctx.fillRect(x0, Ya(cnt[k]), bw, a.y + a.h - Ya(cnt[k]));
      ctx.fillStyle = C.edge; ctx.fillRect(x0, Ya(cnt[k] - cut[k]), bw, a.y + a.h - Ya(cnt[k] - cut[k]));
    }
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "center";
    [0, 0.5, 1].forEach((v) => ctx.fillText(v.toFixed(1), Xa(v), a.y + a.h + 13));
    ctx.textAlign = "right";
    [0, cmax].forEach((c) => ctx.fillText(String(c), a.x - 4, Ya(c) + 3));
    ctx.textAlign = "start";
    if (t <= 1) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(Xa(t), a.y); ctx.lineTo(Xa(t), a.y + a.h); ctx.stroke();
      ctx.font = "600 11px JetBrains Mono, monospace"; ctx.fillStyle = C.ink; ctx.textAlign = "right";
      ctx.fillText(`t = ${t.toFixed(2)}`, Xa(t) - 5, a.y + 14); ctx.textAlign = "start";
    }

    // MAE 곡선: 가로축은 임계값(오른쪽 끝 = 제거 없음)
    let hi = 0; for (const c of curve) hi = Math.max(hi, c.nn || 0, c.lin || 0);
    hi = Math.ceil(hi * 2 + 0.3) / 2;
    const Xb = (v: number) => b.x + ((v - TMIN) / (OFF - TMIN)) * b.w, Yb = (v: number) => b.y + b.h - (v / hi) * b.h;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (let v = 0; v <= hi + 1e-9; v += hi > 2.5 ? 1 : 0.5) { ctx.fillText(v.toFixed(1), b.x - 4, Yb(v) + 3); ctx.beginPath(); ctx.moveTo(b.x, Yb(v)); ctx.lineTo(b.x + b.w, Yb(v)); ctx.stroke(); }
    ctx.textAlign = "center";
    [0.6, 0.8, 1].forEach((v) => ctx.fillText(String(v), Xb(v), b.y + b.h + 13));
    ctx.textAlign = "right";
    ctx.fillText("없음", Xb(OFF), b.y + b.h + 13);
    ctx.textAlign = "start";
    ctx.strokeStyle = C.mid; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(Xb(t), b.y); ctx.lineTo(Xb(t), b.y + b.h); ctx.stroke(); ctx.setLineDash([]);
    const cur = atThreshold(d, ev, t);
    for (const [key, c] of [["nn", C.edge], ["lin", C.ink2]] as ["nn" | "lin", string][]) {
      ctx.strokeStyle = c; ctx.lineWidth = 2.4; ctx.beginPath();
      curve.forEach((p, i) => { const q: [number, number] = [Xb(T_GRID[i]), Yb(p[key])]; i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
      ctx.stroke();
      ctx.fillStyle = c; ctx.strokeStyle = C.bg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(Xb(t), Yb(cur[key]), 5, 0, 7); ctx.fill(); ctx.stroke();
    }
    // 범례(아래 왼쪽: 곡선은 위쪽에 있다)
    ctx.font = "600 11px JetBrains Mono, monospace";
    [["1-NN 암기", C.edge], ["선형 ridge", C.ink2]].forEach(([name, c], i) => {
      const y = b.y + b.h - 26 + i * 16;
      ctx.strokeStyle = c; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(b.x + 8, y - 4); ctx.lineTo(b.x + 24, y - 4); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.fillText(name, b.x + 30, y);
    });
  }

  let raf = 0;
  function update() {
    const c = atThreshold(d, ev, t);
    ro.set("n", `남은 test ${c.n}/${d.te.length}`);
    ro.set("nn", `1-NN MAE ${c.nn.toFixed(2)}`, c.nn < c.lin ? "hot" : "");
    ro.set("lin", `선형 MAE ${c.lin.toFixed(2)}`, c.lin <= c.nn ? "ok" : "");
    ro.set("ratio", `1-NN 겉보기 대비 ${(c.nn / full.nn).toFixed(2)}배`);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pt = param({
    label: "임계", min: TMIN, max: OFF, step: 0.01, value: OFF, fmt: (v) => (v > 1 ? "없음" : v.toFixed(2)),
    aria: "근접 중복으로 보고 test에서 뺄 최근접 유사도 임계값(오른쪽 끝은 제거 없음)", onInput: (v) => { t = v; update(); },
  });
  const grid = el("div", "param-grid");
  grid.append(pt.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
