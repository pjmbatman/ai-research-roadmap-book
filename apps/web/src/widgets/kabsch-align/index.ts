// Kabsch 정렬과 RMSD (data/widgets.toml: kabsch-align).
// 29.3장의 합성 구조: 이상적인 α-나선 두 개(Cα 20개씩, 반지름 2.3 Å, 잔기당 1.5 Å 상승·100° 회전)를 고리 3개로 이은 43잔기.
// 예측 구조 = 기준 구조에 (1) 경첩: 잔기 20부터를 고리 시작점 X[20]을 지나는 y축 둘레로 θ 회전, (2) 좌표마다 σ Å 잡음(고정 시드),
// (3) 강체 운동: 무게중심 둘레로 φ 회전하고 평행이동. 강체 운동은 정렬 후 지표를 바꾸지 않는다.
// 최적 회전은 Horn의 사원수 방법(4×4 대칭 행렬의 최대 고유벡터)으로 구한다. 반사 없는 최적 회전이라 장의 Kabsch(d = sign det)와 같은 해다.
// TM-score는 장과 같은 근사(길이 8 구간마다 중첩해 본 최댓값, 실제 값의 하한), lDDT는 Cα만 쓴 단순화판(15 Å, 허용 오차 0.5·1·2·4 Å).
import { frame, param, presets, readouts, el, C, alpha, rng } from "../frame";
import { pageState } from "../../client/store";

type V3 = [number, number, number];
type M3 = [V3, V3, V3];

export function reference(): V3[] {
  const helix = (n: number, start: V3, sign: number): V3[] => Array.from({ length: n }, (_, t) => {
    const a = (100 * Math.PI / 180) * t;
    return [2.3 * Math.cos(a) + start[0], 2.3 * Math.sin(a) + start[1], 1.5 * t * sign + start[2]];
  });
  const h1 = helix(20, [0, 0, 0], 1), h2 = helix(20, [10, 0, 28.5], -1);
  const loop: V3[] = [1, 2, 3].map((k) => { const s = k / 4; return [0, 1, 2].map((d) => h1[19][d] + s * (h2[0][d] - h1[19][d])) as V3; });
  return [...h1, ...loop, ...h2];
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (R: M3, v: V3): V3 => [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mean = (P: V3[]): V3 => { const c: V3 = [0, 0, 0]; for (const p of P) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; } return [c[0] / P.length, c[1] / P.length, c[2] / P.length]; };
function axisAngle(u: V3, ang: number): M3 {
  const n = Math.hypot(...u), [x, y, z] = [u[0] / n, u[1] / n, u[2] / n], c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
}

/** 4×4 대칭 행렬의 고유값 분해(Jacobi). 최대 고유값의 고유벡터를 돌려준다. */
function topEigvec4(A: number[][]): number[] {
  const a = A.map((r) => r.slice()), v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 4; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 4; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  let best = 0; for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/** P를 Q에 겹치는 최적 회전 R(반사 없음)과 무게중심: R(p − p̄) + q̄ ≈ q */
export function kabsch(P: V3[], Q: V3[]) {
  const pc = mean(P), qc = mean(Q);
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < P.length; i++) { const p = sub(P[i], pc), q = sub(Q[i], qc); for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += p[a] * q[b]; }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  const [q0, q1, q2, q3] = topEigvec4(N);
  const R: M3 = [
    [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
    [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
  ];
  return { R, pc, qc };
}
const applyFit = (P: V3[], f: ReturnType<typeof kabsch>) => P.map((p) => add(mul(f.R, sub(p, f.pc)), f.qc));
export const rmsd = (P: V3[], Q: V3[]) => Math.sqrt(P.reduce((s, p, i) => s + dist(p, Q[i]) ** 2, 0) / P.length);
export const kabschRmsd = (P: V3[], Q: V3[]) => rmsd(applyFit(P, kabsch(P, Q)), Q);

export function tmScore(P: V3[], Q: V3[]) {
  const L = Q.length, d0 = 1.24 * Math.cbrt(L - 15) - 1.8;
  let best = 0;
  for (let s = 0; s + 8 <= L; s++) {
    const fit = kabsch(P.slice(s, s + 8), Q.slice(s, s + 8));
    const A = applyFit(P, fit);
    let t = 0; for (let i = 0; i < L; i++) t += 1 / (1 + (dist(A[i], Q[i]) / d0) ** 2);
    best = Math.max(best, t / L);
  }
  return best;
}

/** 잔기별 단순화 lDDT(Cα)와 전체 값(모든 쌍의 평균) */
export function lddt(P: V3[], Q: V3[], r0 = 15) {
  const L = Q.length, per = new Float64Array(L);
  let tot = 0, cnt = 0;
  for (let i = 0; i < L; i++) {
    let s = 0, c = 0;
    for (let j = 0; j < L; j++) {
      if (i === j) continue;
      const dq = dist(Q[i], Q[j]);
      if (dq >= r0) continue;
      const e = Math.abs(dist(P[i], P[j]) - dq);
      const sc = ((e < 0.5 ? 1 : 0) + (e < 1 ? 1 : 0) + (e < 2 ? 1 : 0) + (e < 4 ? 1 : 0)) / 4;
      s += sc; c++;
    }
    per[i] = c ? s / c : 1; tot += s; cnt += c;
  }
  return { per, all: tot / cnt };
}

/** 예측 구조: 경첩 θ(도), 잡음 σ(Å), 강체 회전 φ(도) */
export function model(X: V3[], Z: V3[], theta: number, sigma: number, phi: number): V3[] {
  const th = (theta * Math.PI) / 180;
  const Ry: M3 = [[Math.cos(th), 0, Math.sin(th)], [0, 1, 0], [-Math.sin(th), 0, Math.cos(th)]];
  let P = X.map((x, i) => (i >= 20 ? add(mul(Ry, sub(x, X[20])), X[20]) : x));
  P = P.map((p, i) => add(p, [sigma * Z[i][0], sigma * Z[i][1], sigma * Z[i][2]]));
  const c = mean(P), Rg = axisAngle([1, 1, 0.5], (phi * Math.PI) / 180), sh = phi / 180;
  return P.map((p) => add(add(mul(Rg, sub(p, c)), c), [10 * sh, -6 * sh, 4 * sh]));
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "Kabsch 정렬과 RMSD", "manipulate", narrow ? 0.95 : 0.6, "기준 구조와 예측 구조의 Cα 사슬을 겹쳐 그린 그림과 잔기별 거리·lDDT 막대");
  const X = reference(), Lr = X.length;
  const r = rng(29);
  const Z: V3[] = X.map(() => [r.normal(), r.normal(), r.normal()]);
  let theta = 35, sigma = 0, phi = 60, aligned = true;
  let P = model(X, Z, theta, sigma, phi), A = P, per = new Float64Array(Lr), dists = new Float64Array(Lr);
  // 보는 방향: 나선 축(z)을 화면 가로로, 두 나선이 위아래로 보이게 x·y를 섞어 세로로
  const cb = Math.cos(0.5), sb = Math.sin(0.5);
  const view = (v: V3): [number, number] => [v[2], v[0] * cb + v[1] * sb];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const stripH = Math.round(H * 0.26), top = 44;
    const vh = H - stripH - 40 - top;
    const shown = aligned ? A : P;
    // 기준과 보이는 예측 구조를 함께 담도록 축척을 맞춘다
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of [...X, ...shown]) { const [vx, vy] = view(p); x0 = Math.min(x0, vx); x1 = Math.max(x1, vx); y0 = Math.min(y0, vy); y1 = Math.max(y1, vy); }
    const sc = Math.min((W - 40) / Math.max(x1 - x0, 1), vh / Math.max(y1 - y0, 1), 14);
    const ox = W / 2 - ((x0 + x1) / 2) * sc, oy = top + vh / 2 + ((y0 + y1) / 2) * sc;
    const proj = (p: V3): [number, number] => { const v = view(p); return [ox + v[0] * sc, oy - v[1] * sc]; };
    const chain = (S: V3[], col: string, lw: number, dotR: number) => {
      const pts = S.map(proj);
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = "round";
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
      ctx.fillStyle = col; for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, dotR, 0, 7); ctx.fill(); }
      return pts;
    };
    // 대응 잔기를 잇는 선(어긋남)
    const pr = X.map(proj), pm = shown.map(proj);
    ctx.strokeStyle = alpha(C.edge, 0.35); ctx.lineWidth = 1;
    for (let i = 0; i < Lr; i++) { ctx.beginPath(); ctx.moveTo(pr[i][0], pr[i][1]); ctx.lineTo(pm[i][0], pm[i][1]); ctx.stroke(); }
    chain(X, C.mid, 3, 2.2);
    chain(shown, C.edge, 2.2, 2.4);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "left";
    ctx.strokeStyle = C.mid; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(10, 16); ctx.lineTo(26, 16); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.fillText("기준(실험) 구조", 32, 20);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(10, 32); ctx.lineTo(26, 32); ctx.stroke();
    ctx.fillText(aligned ? "예측 구조(Kabsch 정렬 후)" : "예측 구조(정렬 전)", 32, 36);
    // ── 잔기별 거리와 lDDT
    const B = { x: 40, y: H - stripH - 14, w: W - 52, h: stripH - 16 };
    const bw = B.w / Lr, dmax = 10;
    const by = (v: number) => B.y + B.h - (Math.min(v, dmax) / dmax) * (B.h - 10);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(B.x + 0.5, B.y + 0.5, B.w - 1, B.h - 1);
    for (let i = 0; i < Lr; i++) {
      ctx.fillStyle = dists[i] > 4 ? C.edge : alpha(C.ink2, 0.8);
      ctx.fillRect(B.x + i * bw + 1, by(dists[i]), Math.max(1, bw - 2), B.y + B.h - by(dists[i]));
      ctx.fillStyle = alpha(C.good, 0.15 + 0.85 * per[i]);
      ctx.fillRect(B.x + i * bw + 1, B.y + B.h + 3, Math.max(1, bw - 2), 6);
    }
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "right";
    for (const v of [0, 5, 10]) ctx.fillText(String(v), B.x - 5, by(v) + 4);
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(aligned ? "잔기별 거리 dᵢ (Å) · 아래 띠: 잔기별 lDDT" : "잔기별 거리 dᵢ (Å, 정렬 전) · 아래 띠: 잔기별 lDDT", B.x, B.y - 7);
    // 나선·고리 구간 표시
    ctx.fillStyle = alpha(C.mid, 0.12); ctx.fillRect(B.x + 20 * bw, B.y + 1, 3 * bw, B.h - 2);
  }

  function update() {
    P = model(X, Z, theta, sigma, phi);
    A = applyFit(P, kabsch(P, X));
    const shown = aligned ? A : P;
    for (let i = 0; i < Lr; i++) dists[i] = dist(shown[i], X[i]);
    const ld = lddt(P, X);
    per = ld.per;
    ro.set("k", `RMSD 정렬 전 ${rmsd(P, X).toFixed(2)} → 후 ${rmsd(A, X).toFixed(2)} Å`, "hot");
    ro.set("loc", `TM ${tmScore(P, X).toFixed(2)} · lDDT ${ld.all.toFixed(2)}`, "ok");
    pageState.set("kabsch", { hinge_deg: theta, noise_A: sigma, rigid_deg: phi, rmsd_raw_A: rmsd(P, X), rmsd_kabsch_A: rmsd(A, X), tm: tmScore(P, X), lddt: ld.all }, root);
    draw();
  }

  const pPhi = param({ label: "회전", min: 0, max: 180, step: 5, value: phi, fmt: (v) => `${v}°`, aria: "예측 구조 전체의 회전 각도(평행이동 포함)", onInput: (v) => { phi = v; pre.select(null); update(); } });
  const pTh = param({ label: "경첩", min: 0, max: 90, step: 5, value: theta, fmt: (v) => `${v}°`, aria: "두 번째 나선의 경첩 회전 각도", onInput: (v) => { theta = v; pre.select(null); update(); } });
  const pS = param({ label: "잡음", min: 0, max: 3, step: 0.1, value: sigma, fmt: (v) => `${v.toFixed(1)} Å`, aria: "좌표마다 더하는 잡음의 표준편차 (옹스트롬)", onInput: (v) => { sigma = v; pre.select(null); update(); } });
  const grid = el("div", "param-grid");
  grid.append(pPhi.row, pTh.row, pS.row);
  type K = "hinge" | "noise" | "rigid";
  const PV: Record<K, [number, number, number]> = { hinge: [60, 35, 0], noise: [60, 0, 1.3], rigid: [120, 0, 0.5] };
  for (const p of [pPhi, pTh, pS]) p.row.style.gridTemplateColumns = "2.6em minmax(0, 1fr) 3.4em";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(170px, 1fr))";
  const pre = presets<K>([{ key: "hinge", label: "경첩 35°" }, { key: "noise", label: "잡음 1.3 Å" }, { key: "rigid", label: "강체 + 잡음 0.5 Å" }], (k) => {
    [phi, theta, sigma] = PV[k]; pPhi.set(phi); pTh.set(theta); pS.set(sigma); update();
  }, "hinge");
  pre.row.classList.add("segmented");
  if (narrow) pre.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b) => (b.style.padding = "0 9px"));
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" checked /> Kabsch 정렬해서 보기`;
  sw.querySelector("input")!.addEventListener("change", (ev) => { aligned = (ev.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
