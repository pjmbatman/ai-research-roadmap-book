// 배치 효과와 보정 (data/widgets.toml: batch-correction).
// 29.4장 배치 효과 절의 실험을 로그 공간의 작은 합성 데이터로 줄였다(개수 분포·정규화 단계는 생략한 단순화).
//   유전자 60개: 유형 t의 marker 8개(8t..8t+7)는 +1.6, 반응 유전자 8개(24..31)는 배치 2의 유형 2에서만 +1.2(처리 반응),
//   나머지는 배경. 세포 잡음 N(0, 0.45²). 배치 1은 유형마다 100개, 배치 2는 200개로 구성이 교락 c에 따라 바뀐다:
//   유형 비율 ((1−c)/3, 1/3, (1+c)/3). 배치 2의 모든 세포에 유전자마다 δ_g = s·z_g가 더해진다(z_g ~ N(0, 1) 고정).
// 보정: 없음 / 배치마다 전체 평균 빼기 / 배치마다 공유 유형 1의 평균 빼기(장과 같은 center 함수).
// 평가: PCA 10차원 위 k-means(k = 3, 초기값 8번 중 관성 최소)의 ARI(유형)·ARI(배치), 반응 유전자 점수의 배치 간 차이(유형 0, 2).
// 참값은 배치 효과를 넣기 전 데이터의 같은 지표다.
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

const G = 60, N1 = 300, N2 = 200, N = N1 + N2, NPC = 10;
export type Method = "none" | "mean" | "shared";

export function makeBase(seed = 29) {
  const r = rng(seed);
  const base = Float64Array.from({ length: G }, () => 1 + 0.5 * r.normal());
  const z = Float64Array.from({ length: G }, () => r.normal());
  const noise = Float64Array.from({ length: N * G }, () => 0.45 * r.normal());
  const u = Float64Array.from({ length: N2 }, () => r.u());
  return { base, z, noise, u };
}
type Base = ReturnType<typeof makeBase>;

export function simulate(b: Base, c: number, s: number) {
  const T = new Int32Array(N), B = new Int32Array(N);
  for (let i = 0; i < N1; i++) T[i] = Math.floor(i / 100);
  const p0 = (1 - c) / 3, p1 = 1 / 3;
  for (let i = 0; i < N2; i++) { const v = b.u[i]; T[N1 + i] = v < p0 ? 0 : v < p0 + p1 ? 1 : 2; B[N1 + i] = 1; }
  const truth = new Float64Array(N * G), M = new Float64Array(N * G);
  for (let i = 0; i < N; i++) for (let g = 0; g < G; g++) {
    let v = b.base[g] + b.noise[i * G + g];
    if (g >= 8 * T[i] && g < 8 * T[i] + 8) v += 1.6;
    if (g >= 24 && g < 32 && T[i] === 2 && B[i] === 1) v += 1.2;
    truth[i * G + g] = v;
    M[i * G + g] = v + (B[i] === 1 ? s * b.z[g] : 0);
  }
  return { T, B, truth, M };
}

export function correct(M: Float64Array, T: Int32Array, B: Int32Array, method: Method) {
  if (method === "none") return M;
  const out = M.slice();
  for (const bb of [0, 1]) {
    const mu = new Float64Array(G); let n = 0;
    for (let i = 0; i < N; i++) if (B[i] === bb && (method === "mean" || T[i] === 1)) { n++; for (let g = 0; g < G; g++) mu[g] += M[i * G + g]; }
    for (let g = 0; g < G; g++) mu[g] /= Math.max(1, n);
    for (let i = 0; i < N; i++) if (B[i] === bb) for (let g = 0; g < G; g++) out[i * G + g] -= mu[g];
  }
  return out;
}

/** 대칭 행렬의 고유값 분해(순환 Jacobi). 고유값 내림차순 */
function eigSym(A: Float64Array, n: number) {
  const a = A.slice(), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-14) continue;
      const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < n; k++) { const kp = a[k * n + p], kq = a[k * n + q]; a[k * n + p] = cs * kp - sn * kq; a[k * n + q] = sn * kp + cs * kq; }
      for (let k = 0; k < n; k++) { const pk = a[p * n + k], qk = a[q * n + k]; a[p * n + k] = cs * pk - sn * qk; a[q * n + k] = sn * pk + cs * qk; }
      for (let k = 0; k < n; k++) { const kp = v[k * n + p], kq = v[k * n + q]; v[k * n + p] = cs * kp - sn * kq; v[k * n + q] = sn * kp + cs * kq; }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[y * n + y] - a[x * n + x]);
  return { vals: idx.map((i) => a[i * n + i]), vecs: idx.map((i) => Float64Array.from({ length: n }, (_, k) => v[k * n + i])) };
}

/** 중심화한 뒤 PCA 점수 (N × NPC) */
export function pca(X: Float64Array) {
  const mu = new Float64Array(G);
  for (let i = 0; i < N; i++) for (let g = 0; g < G; g++) mu[g] += X[i * G + g] / N;
  const cov = new Float64Array(G * G);
  for (let i = 0; i < N; i++) for (let a = 0; a < G; a++) { const xa = X[i * G + a] - mu[a]; for (let b = a; b < G; b++) cov[a * G + b] += xa * (X[i * G + b] - mu[b]); }
  for (let a = 0; a < G; a++) for (let b = a; b < G; b++) cov[b * G + a] = cov[a * G + b] /= N - 1;
  const { vecs } = eigSym(cov, G);
  const Z = new Float64Array(N * NPC);
  for (let i = 0; i < N; i++) for (let k = 0; k < NPC; k++) { let sm = 0; for (let g = 0; g < G; g++) sm += (X[i * G + g] - mu[g]) * vecs[k][g]; Z[i * NPC + k] = sm; }
  return Z;
}

export function kmeans(Z: Float64Array, k = 3, inits = 8) {
  let best: Int32Array = new Int32Array(N), bestIn = Infinity;
  for (let s = 0; s < inits; s++) {
    const r = rng(500 + s);
    // k-means++ 초기화
    const cent: number[][] = [];
    const first = Math.floor(r.u() * N);
    cent.push(Array.from(Z.subarray(first * NPC, first * NPC + NPC)));
    const d2 = new Float64Array(N);
    while (cent.length < k) {
      let tot = 0;
      for (let i = 0; i < N; i++) { let m = Infinity; for (const c of cent) { let dd = 0; for (let j = 0; j < NPC; j++) dd += (Z[i * NPC + j] - c[j]) ** 2; m = Math.min(m, dd); } d2[i] = m; tot += m; }
      let t = r.u() * tot, pick = N - 1;
      for (let i = 0; i < N; i++) { t -= d2[i]; if (t <= 0) { pick = i; break; } }
      cent.push(Array.from(Z.subarray(pick * NPC, pick * NPC + NPC)));
    }
    const lab = new Int32Array(N);
    let inertia = 0;
    for (let it = 0; it < 100; it++) {
      let changed = false; inertia = 0;
      for (let i = 0; i < N; i++) {
        let m = Infinity, bi = 0;
        for (let c = 0; c < k; c++) { let dd = 0; for (let j = 0; j < NPC; j++) dd += (Z[i * NPC + j] - cent[c][j]) ** 2; if (dd < m) { m = dd; bi = c; } }
        if (lab[i] !== bi) { changed = true; lab[i] = bi; }
        inertia += m;
      }
      for (let c = 0; c < k; c++) { const sm = new Array(NPC).fill(0); let n = 0; for (let i = 0; i < N; i++) if (lab[i] === c) { n++; for (let j = 0; j < NPC; j++) sm[j] += Z[i * NPC + j]; } if (n) cent[c] = sm.map((v) => v / n); }
      if (!changed && it > 0) break;
    }
    if (inertia < bestIn) { bestIn = inertia; best = lab.slice(); }
  }
  return best;
}

export function ari(a: Int32Array, b: Int32Array) {
  const ka = Math.max(...a) + 1, kb = Math.max(...b) + 1, tab = new Float64Array(ka * kb), ra = new Float64Array(ka), rb = new Float64Array(kb);
  for (let i = 0; i < a.length; i++) { tab[a[i] * kb + b[i]]++; ra[a[i]]++; rb[b[i]]++; }
  const c2 = (x: number) => (x * (x - 1)) / 2;
  let sij = 0; for (const v of tab) sij += c2(v);
  let sa = 0; for (const v of ra) sa += c2(v);
  let sb = 0; for (const v of rb) sb += c2(v);
  const exp = (sa * sb) / c2(a.length), mx = (sa + sb) / 2;
  return mx === exp ? 0 : (sij - exp) / (mx - exp);
}

export function response(X: Float64Array, T: Int32Array, B: Int32Array, t: number) {
  let s1 = 0, n1 = 0, s0 = 0, n0 = 0;
  for (let i = 0; i < N; i++) {
    if (T[i] !== t) continue;
    let r = 0; for (let g = 24; g < 32; g++) r += X[i * G + g] / 8;
    if (B[i] === 1) { s1 += r; n1++; } else { s0 += r; n0++; }
  }
  return n1 && n0 ? s1 / n1 - s0 / n0 : NaN;
}

export function evaluate(b: Base, c: number, s: number, method: Method) {
  const sim = simulate(b, c, s);
  const X = correct(sim.M, sim.T, sim.B, method);
  const Z = pca(X), lab = kmeans(Z);
  return { sim, Z, ariType: ari(sim.T, lab), ariBatch: ari(sim.B, lab), r0: response(X, sim.T, sim.B, 0), r2: response(X, sim.T, sim.B, 2), t0: response(sim.truth, sim.T, sim.B, 0), t2: response(sim.truth, sim.T, sim.B, 2) };
}

export function mount(root: HTMLElement) {
  const wide = (root.clientWidth || 600) >= 640;
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "배치 효과와 보정", "simulate", wide ? 0.56 : narrow ? 0.98 : 0.62, "세포를 PCA 첫 두 주성분에 찍은 산점도(색 = 세포 유형, 속이 빈 점 = 배치 2)와 반응 유전자 차이 막대");
  const b = makeBase();
  let c = 0.4, s = 0.8, method: Method = "none";
  let ev = evaluate(b, c, s, method);
  const TYPE_COL = () => [C.ink2, C.good, C.edge];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const mono = "11px JetBrains Mono, monospace", sans = "600 12px Pretendard Variable, sans-serif";
    type Box = { x: number; y: number; w: number; h: number };
    let A: Box, Bx: Box;
    if (wide) { const lw = Math.round(W * 0.6); A = { x: 14, y: 26, w: lw - 24, h: H - 40 }; Bx = { x: lw + 56, y: 40, w: W - lw - 56 - 16, h: H - 80 }; }
    else { const bh = Math.round(H * 0.24); A = { x: 14, y: 26, w: W - 28, h: H - bh - 52 }; Bx = { x: 56, y: H - bh - 4, w: W - 56 - 16, h: bh - 14 }; }
    // ── PCA 산점도: 부호를 유형 평균으로 고정
    const { Z, sim } = ev;
    const m1 = [0, 0, 0], m2 = [0, 0, 0], cnt = [0, 0, 0];
    for (let i = 0; i < N; i++) { m1[sim.T[i]] += Z[i * NPC]; m2[sim.T[i]] += Z[i * NPC + 1]; cnt[sim.T[i]]++; }
    const sx = m1[0] / Math.max(1, cnt[0]) - m1[2] / Math.max(1, cnt[2]) > 0 ? -1 : 1;
    const sy = m2[1] / Math.max(1, cnt[1]) > (m2[0] / Math.max(1, cnt[0]) + m2[2] / Math.max(1, cnt[2])) / 2 ? 1 : -1;
    let ext = 0;
    for (let i = 0; i < N; i++) ext = Math.max(ext, Math.abs(Z[i * NPC]), Math.abs(Z[i * NPC + 1]));
    ext *= 1.06;
    const sc = Math.min(A.w, A.h) / (2 * ext), cx = A.x + A.w / 2, cy = A.y + A.h / 2;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(A.x, cy); ctx.lineTo(A.x + A.w, cy); ctx.moveTo(cx, A.y); ctx.lineTo(cx, A.y + A.h); ctx.stroke();
    const cols = TYPE_COL();
    for (let pass = 0; pass < 2; pass++) for (let i = 0; i < N; i++) {
      if (sim.B[i] !== pass) continue;
      const x = cx + sx * Z[i * NPC] * sc, y = cy - sy * Z[i * NPC + 1] * sc, col = cols[sim.T[i]];
      if (sim.B[i] === 0) { ctx.fillStyle = alpha(col, 0.7); ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.fill(); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 7); ctx.stroke(); }
    }
    ctx.font = sans; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    ctx.fillText("PC1 × PC2", A.x, A.y - 9);
    // 범례(제목 줄 오른쪽): 유형 색, 배치 모양
    ctx.font = mono; ctx.textAlign = "left";
    const items: [string, (x: number, y: number) => void][] = [
      ...[0, 1, 2].map((k) => [`유형${k}`, (x: number, y: number) => { ctx.fillStyle = cols[k]; ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill(); }] as [string, (x: number, y: number) => void]),
      ["배치2", (x: number, y: number) => { ctx.strokeStyle = C.mid; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 7); ctx.stroke(); }],
    ];
    const widths = items.map(([t]) => ctx.measureText(t).width + 18);
    let lx = A.x + A.w - widths.reduce((p, q) => p + q, 0);
    items.forEach(([t, mark], k) => { mark(lx + 4, A.y - 13); ctx.fillStyle = C.mid; ctx.fillText(t, lx + 11, A.y - 9); lx += widths[k]; });
    // ── 반응 차이 막대
    const lo = -1.6, hi = 1.6;
    const X = (v: number) => Bx.x + ((v - lo) / (hi - lo)) * Bx.w;
    ctx.font = sans; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    ctx.fillText("반응 유전자 차이(배치 2 − 1)", wide ? Bx.x - 42 : Bx.x - 42, Bx.y - 10);
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(0), Bx.y); ctx.lineTo(X(0), Bx.y + Bx.h); ctx.stroke();
    ctx.font = mono; ctx.textAlign = "center"; ctx.fillStyle = C.mid;
    for (const v of [-1, 0, 1]) ctx.fillText((v > 0 ? "+" : "") + v, X(v), Bx.y + Bx.h + 13);
    const rows: [string, number, number, string][] = [["유형 0", ev.r0, ev.t0, cols[0]], ["유형 2", ev.r2, ev.t2, cols[2]]];
    const rh = Bx.h / rows.length;
    rows.forEach(([name, v, t, col], k) => {
      const y0 = Bx.y + k * rh + rh * 0.22, bh = rh * 0.56;
      ctx.textAlign = "right"; ctx.fillStyle = C.mid; ctx.fillText(name, Bx.x - 6, y0 + bh / 2 + 4);
      if (Number.isFinite(v)) { ctx.fillStyle = alpha(col, 0.75); ctx.fillRect(Math.min(X(0), X(v)), y0, Math.abs(X(v) - X(0)), bh); }
      if (Number.isFinite(t)) { ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.setLineDash([3, 2]); ctx.beginPath(); ctx.moveTo(X(t), y0 - 4); ctx.lineTo(X(t), y0 + bh + 4); ctx.stroke(); ctx.setLineDash([]); }
    });
    ctx.textAlign = "right"; ctx.fillStyle = C.mid;
    ctx.fillText("점선: 참값", Bx.x + Bx.w, Bx.y - 10 + (wide ? 0 : 0));
  }

  let raf = 0;
  function update() {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(compute);
  }
  function compute() {
    ev = evaluate(b, c, s, method);
    const fmt = (v: number) => (Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(2) : "—");
    ro.set("ari", `ARI 유형 ${ev.ariType.toFixed(2)} · 배치 ${ev.ariBatch.toFixed(2)}`, ev.ariBatch > 0.3 ? "hot" : "");
    ro.set("r2", `반응 유형2 ${fmt(ev.r2)}/참 ${fmt(ev.t2)}`, Math.abs(ev.r2 - ev.t2) > 0.2 ? "hot" : "ok");
    ro.set("r0", `유형0 ${fmt(ev.r0)}/참 ${fmt(ev.t0)}`, Math.abs(ev.r0 - ev.t0) > 0.2 ? "hot" : "ok");
    pageState.set("batchCorrection", { batch_effect: s, confound: c, method, ari_type: ev.ariType, ari_batch: ev.ariBatch, response_type2: ev.r2, response_type0: ev.r0, truth_type2: ev.t2, truth_type0: ev.t0 }, root);
    draw();
  }

  const pS = param({ label: "배치 s", min: 0, max: 1.5, step: 0.05, value: s, fmt: (v) => v.toFixed(2), aria: "배치 2에 더하는 유전자별 배치 효과의 크기", onInput: (v) => { s = v; update(); } });
  const pC = param({ label: "교락 c", min: 0, max: 0.9, step: 0.05, value: c, fmt: (v) => v.toFixed(2), aria: "배치 2의 세포 구성이 배치 1과 다른 정도", onInput: (v) => { c = v; update(); } });
  for (const p of [pS, pC]) p.row.style.gridTemplateColumns = "3.6em minmax(0, 1fr) 2.8em";
  const grid = el("div", "param-grid");
  grid.append(pS.row, pC.row);
  const pre = presets<Method>([{ key: "none", label: "보정 없음" }, { key: "mean", label: "배치 평균 빼기" }, { key: "shared", label: "공유 유형 1 기준" }], (k) => { method = k; update(); }, method);
  pre.row.classList.add("segmented");
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((bt) => (bt.style.padding = "0 9px"));
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  compute();
}
