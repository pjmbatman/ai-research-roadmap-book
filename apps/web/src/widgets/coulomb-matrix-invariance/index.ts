// Coulomb 행렬 고윳값의 불변성 (data/widgets.toml: coulomb-matrix-invariance).
// 작은 분자(좌표는 합성)의 Coulomb 행렬 M_ii = ½ Z_i^2.4, M_ij = Z_i Z_j / |R_i − R_j|(bohr)과 정렬된 고윳값을 함께 그린다.
// 회전·이동은 좌표 배열을 바꾸지만 거리는 그대로라 M도 그대로다. 원자 번호를 바꿔 매기면 M이 P M Pᵀ로 바뀌지만
// 고윳값은 그대로다. 결합 하나를 늘이면(분자가 실제로 달라지면) 고윳값도 달라진다.
// 고윳값은 대칭 행렬의 Jacobi 회전으로 직접 구한다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

type V3 = [number, number, number];
export const ANG_TO_BOHR = 1.8897259886;
interface Mol { label: string; sym: string[]; Z: number[]; R: V3[]; bonds: [number, number][] }
const s3 = 1.09 / Math.sqrt(3);
export const MOLS: Record<string, Mol> = {
  water: {
    label: "물 H₂O", sym: ["O", "H", "H"], Z: [8, 1, 1],
    R: [[0, 0, 0], [0.7572, 0.5865, 0], [-0.7572, 0.5865, 0]], bonds: [[0, 1], [0, 2]],
  },
  formaldehyde: {
    label: "폼알데하이드 CH₂O", sym: ["C", "O", "H", "H"], Z: [6, 8, 1, 1],
    R: [[0, 0, 0], [0, 1.21, 0], [0.94, -0.59, 0], [-0.94, -0.59, 0]], bonds: [[0, 1], [0, 2], [0, 3]],
  },
  methane: {
    label: "메테인 CH₄", sym: ["C", "H", "H", "H", "H"], Z: [6, 1, 1, 1, 1],
    R: [[0, 0, 0], [s3, s3, s3], [s3, -s3, -s3], [-s3, s3, -s3], [-s3, -s3, s3]], bonds: [[0, 1], [0, 2], [0, 3], [0, 4]],
  },
};

/** Coulomb 행렬. R은 bohr. */
export function coulomb(Z: number[], R: V3[]): number[][] {
  const n = Z.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
    if (i === j) return 0.5 * Z[i] ** 2.4;
    return (Z[i] * Z[j]) / Math.hypot(R[i][0] - R[j][0], R[i][1] - R[j][1], R[i][2] - R[j][2]);
  }));
}

/** 대칭 행렬의 고윳값(순환 Jacobi 회전). 큰 것부터 정렬해 돌려준다. */
export function eigSym(M: number[][]): number[] {
  const n = M.length;
  const a = M.map((r) => [...r]);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      // a_pq를 0으로 만드는 회전각: tan 2θ = 2a_pq / (a_qq − a_pp)
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const kp = a[k][p], kq = a[k][q];
        a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq;
      }
      for (let k = 0; k < n; k++) {
        const pk = a[p][k], qk = a[q][k];
        a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk;
      }
    }
  }
  return a.map((r, i) => r[i]).sort((x, y) => y - x);
}

/** 단위 축 u 둘레로 θ만큼 회전(Rodrigues) */
function rotate(p: V3, u: V3, th: number): V3 {
  const c = Math.cos(th), s = Math.sin(th);
  const d = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
  const x: V3 = [u[1] * p[2] - u[2] * p[1], u[2] * p[0] - u[0] * p[2], u[0] * p[1] - u[1] * p[0]];
  return [0, 1, 2].map((k) => p[k] * c + x[k] * s + u[k] * d * (1 - c)) as V3;
}
const norm = (v: V3): V3 => { const l = Math.hypot(...v); return [v[0] / l, v[1] / l, v[2] / l]; };
const AXIS = norm([0.35, 1, 0.55]);
const SHIFT = norm([1, -0.45, 0.3]);

/** 번호 매기기 목록: 순환 이동과 그 역순(2n가지). 첫 번째가 원래 순서다. */
function perms(n: number): number[][] {
  const out: number[][] = [];
  for (let k = 0; k < n; k++) {
    const p = Array.from({ length: n }, (_, i) => (i + k) % n);
    out.push(p, [...p].reverse());
  }
  return out;
}

export interface State { mol: Mol; theta: number; shift: number; stretch: number; perm: number[] }
/** 지금 상태의 원자 배열(새 번호 순서)과 Å 좌표 */
export function configure(s: State) {
  const { mol } = s;
  const base = mol.R.map((r, i) => {
    if (i !== 1) return r;
    // 1번 원자(원래 번호)의 결합을 0번 원자 기준으로 늘인다
    const o = mol.R[0];
    return [0, 1, 2].map((k) => o[k] + (r[k] - o[k]) * s.stretch) as V3;
  });
  const moved = base.map((r) => { const q = rotate(r, AXIS, s.theta); return [0, 1, 2].map((k) => q[k] + SHIFT[k] * s.shift) as V3; });
  const R = s.perm.map((i) => moved[i]);
  const Z = s.perm.map((i) => mol.Z[i]);
  return { R, Z, sym: s.perm.map((i) => mol.sym[i]) };
}
export function analyze(s: State) {
  const ref = coulomb(s.mol.Z, s.mol.R.map((r) => r.map((v) => v * ANG_TO_BOHR) as V3));
  const cur = configure(s);
  const M = coulomb(cur.Z, cur.R.map((r) => r.map((v) => v * ANG_TO_BOHR) as V3));
  const e0 = eigSym(ref), e = eigSym(M);
  let dM = 0, dR = 0;
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M.length; j++) dM += (M[i][j] - ref[i][j]) ** 2;
    dR += (cur.R[i][0] - s.mol.R[i][0]) ** 2 + (cur.R[i][1] - s.mol.R[i][1]) ** 2 + (cur.R[i][2] - s.mol.R[i][2]) ** 2;
  }
  const dE = Math.max(...e.map((v, i) => Math.abs(v - e0[i])));
  return { ...cur, M, ref, e, e0, dM: Math.sqrt(dM), dR: Math.sqrt(dR / M.length), dE };
}

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 560;
  const f = frame(root, "Coulomb 행렬 고윳값의 불변성", "compare", narrow ? 1.12 : 0.44, "분자 배치, 그 Coulomb 행렬, 처음과 지금의 정렬된 고윳값을 나란히 그린 그림");
  let key = attrs.preset in MOLS ? attrs.preset : "water";
  let order = perms(MOLS[key].Z.length), pi = 0;
  const s: State = { mol: MOLS[key], theta: 0, shift: 0, stretch: 1, perm: order[0] };
  let res = analyze(s);

  function boxes() {
    const W = f.width, H = f.height, top = 24, pad = 10, gap = 16;
    if (W >= 560) {
      const w = (W - 2 * pad - 2 * gap) / 3, h = H - top - pad;
      const side = Math.min(w, h);
      return { mol: { x: pad, y: top, w, h: side }, mat: { x: pad + w + gap, y: top, w, h: side }, eig: { x: pad + 2 * (w + gap), y: top, w, h: side } };
    }
    const w = (W - 2 * pad - gap) / 2, side = Math.min(w, H * 0.52);
    const y2 = top + side + 32;
    return { mol: { x: pad, y: top, w, h: side }, mat: { x: pad + w + gap, y: top, w, h: side }, eig: { x: pad, y: y2, w: W - 2 * pad, h: H - y2 - pad } };
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const B = boxes();
    const n = res.Z.length;
    ctx.textBaseline = "alphabetic";
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.mid; ctx.textAlign = "start";
    ctx.fillText("분자 (x, y 투영)", B.mol.x, B.mol.y - 8);
    ctx.fillText("Coulomb 행렬 M", B.mat.x, B.mat.y - 8);
    ctx.fillText("정렬된 고윳값 (로그 축)", B.eig.x, B.eig.y - 8);

    // 분자: 회전·이동 뒤 x, y 투영. 처음 배치는 흐리게 겹쳐 그린다.
    const m = B.mol, sideM = Math.min(m.w, m.h), cx = m.x + m.w / 2, cy = m.y + m.h / 2, sc = sideM / 6;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(m.x, m.y, m.w, m.h);
    const P = (r: V3): [number, number] => [cx + r[0] * sc, cy - r[1] * sc];
    const drawMol = (R: V3[], sym: string[], idx: number[] | null, ghost: boolean) => {
      // 결합(원래 번호 기준)
      const pos = new Map<number, V3>();
      (idx ?? R.map((_, i) => i)).forEach((orig, i) => pos.set(orig, R[i]));
      ctx.strokeStyle = ghost ? C.faint : C.mid; ctx.lineWidth = ghost ? 1.5 : 2.5;
      for (const [a, b] of s.mol.bonds) { ctx.beginPath(); ctx.moveTo(...P(pos.get(a)!)); ctx.lineTo(...P(pos.get(b)!)); ctx.stroke(); }
      const ord = R.map((_, i) => i).sort((a, b) => R[a][2] - R[b][2]);
      const rad = (i: number) => (sym[i] !== "H" ? 11 : 7) * (1 + 0.08 * R[i][2]);
      for (const i of ord) {
        const [px, py] = P(R[i]);
        if (ghost) { ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(px, py, rad(i), 0, 7); ctx.stroke(); continue; }
        ctx.fillStyle = sym[i] !== "H" ? C.edge : C.ink2; ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py, rad(i), 0, 7); ctx.fill(); ctx.stroke();
      }
      if (ghost) return;
      // 번호 라벨: 분자 중심에서 바깥쪽으로 띄운다
      const ctr = R.reduce((a, r) => [a[0] + r[0] / R.length, a[1] + r[1] / R.length], [0, 0]);
      ctx.font = "600 11px JetBrains Mono, monospace"; ctx.textAlign = "center"; ctx.lineJoin = "round";
      R.forEach((r, i) => {
        let dx = r[0] - ctr[0], dy = r[1] - ctr[1];
        const l = Math.hypot(dx, dy);
        if (l < 0.15) { dx = -0.7; dy = 0.7; } else { dx /= l; dy /= l; }
        const [px, py] = P(r), off = rad(i) + 10;
        const tx = px + dx * off, ty = py - dy * off + 4;
        ctx.strokeStyle = C.bg; ctx.lineWidth = 3; ctx.strokeText(`${sym[i]}${i + 1}`, tx, ty);
        ctx.fillStyle = C.ink; ctx.fillText(`${sym[i]}${i + 1}`, tx, ty);
      });
      ctx.textAlign = "start";
    };
    ctx.save(); ctx.beginPath(); ctx.rect(m.x, m.y, m.w, m.h); ctx.clip();
    drawMol(s.mol.R, s.mol.sym, null, true);
    drawMol(res.R, res.sym, s.perm, false);
    ctx.restore();

    // 행렬: 칸 색은 log(1 + |M_ij|), 처음 행렬과 다른 칸은 테두리로 표시
    const t = B.mat, lab = 22, cs = Math.min((t.w - lab) / n, (t.h - lab) / n), mx = t.x + lab, my = t.y + lab;
    const vmax = Math.log1p(Math.max(...res.M.flat()));
    ctx.textAlign = "center";
    const fs = cs >= 40 ? 11 : cs >= 30 ? 10 : 9;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = C.mid; ctx.font = `600 ${fs}px JetBrains Mono, monospace`;
      ctx.fillText(`${res.sym[i]}${i + 1}`, mx + (i + 0.5) * cs, my - 7);
      ctx.textAlign = "right"; ctx.fillText(`${res.sym[i]}${i + 1}`, mx - 4, my + (i + 0.5) * cs + 4); ctx.textAlign = "center";
      for (let j = 0; j < n; j++) {
        const v = res.M[i][j], x = mx + j * cs, y = my + i * cs;
        ctx.fillStyle = alpha(C.edge, 0.08 + 0.72 * (Math.log1p(v) / vmax));
        ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        ctx.fillStyle = C.ink; ctx.font = `${fs}px JetBrains Mono, monospace`;
        ctx.fillText(v >= 10 ? v.toFixed(0) : v.toFixed(2), x + cs / 2, y + cs / 2 + fs / 3);
        if (Math.abs(v - res.ref[i][j]) > 1e-9) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); }
      }
    }
    ctx.textAlign = "start";

    // 고윳값: 처음(테두리 막대)과 지금(채운 막대), 로그 축 0.1 ~ 100
    const g = B.eig, lx = g.x + 28, valW = 58, bw = g.w - 28 - valW;
    const L = (v: number) => lx + (Math.max(-1, Math.log10(Math.max(v, 1e-3))) + 1) / 3 * bw;
    const rowH = Math.min(30, (g.h - 22) / n), bh = Math.min(16, rowH * 0.62);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "center";
    [0.1, 1, 10, 100].forEach((v) => {
      const x = L(v);
      ctx.beginPath(); ctx.moveTo(x, g.y); ctx.lineTo(x, g.y + rowH * n); ctx.stroke();
      ctx.fillText(String(v), x, g.y + rowH * n + 14);
    });
    for (let i = 0; i < n; i++) {
      const y = g.y + i * rowH + (rowH - bh) / 2;
      ctx.textAlign = "start"; ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillText(`λ${i + 1}`, g.x, y + bh / 2 + 4);
      ctx.fillStyle = C.edge; ctx.fillRect(L(0.1), y, L(res.e[i]) - L(0.1), bh);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(L(0.1), y, L(res.e0[i]) - L(0.1), bh);
      ctx.fillStyle = C.ink; ctx.textAlign = "right";
      ctx.fillText(res.e[i].toFixed(3), g.x + g.w, y + bh / 2 + 4);
    }
    ctx.textAlign = "start";
  }

  let raf = 0;
  function update() {
    s.perm = order[pi];
    res = analyze(s);
    const sci = (v: number) => (v < 1e-12 ? "0" : v < 1e-3 ? v.toExponential(1).replace(/-/g, "−") : v.toPrecision(3));
    ro.set("dR", `좌표 배열 변화 ${res.dR.toFixed(2)} Å`);
    ro.set("dM", `행렬 변화 ‖ΔM‖ ${res.dM < 1e-9 ? "0" : res.dM.toFixed(2)}`, res.dM > 1e-9 ? "hot" : "");
    ro.set("dE", `고윳값 변화 max|Δλ| ${sci(res.dE)}`, res.dE > 1e-9 ? "hot" : "ok");
    ro.set("ord", `번호 순서 ${res.sym.join("·")}`);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pre = presets(Object.entries(MOLS).map(([k, v]) => ({ key: k, label: v.label })), (k) => {
    key = k; s.mol = MOLS[k]; order = perms(s.mol.Z.length); pi = 0; update();
  }, key);
  const pt = param({ label: "회전", min: 0, max: 360, step: 1, value: 0, fmt: (v) => `${v.toFixed(0)}°`, aria: "기울어진 축 둘레의 회전각", onInput: (v) => { s.theta = (v * Math.PI) / 180; update(); } });
  const pd = param({ label: "이동", min: -1.5, max: 1.5, step: 0.05, value: 0, fmt: (v) => `${v.toFixed(2)} Å`, aria: "고정된 방향으로의 평행 이동 거리(Å)", onInput: (v) => { s.shift = v; update(); } });
  const pb = param({ label: "결합", min: 1, max: 1.4, step: 0.01, value: 1, fmt: (v) => `×${v.toFixed(2)}`, aria: "원래 2번 원자의 결합 길이 배율", onInput: (v) => { s.stretch = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pt.row, pd.row, pb.row);
  const tools = el("div", "wframe-row");
  const shuffle = el("button", "preset");
  shuffle.type = "button";
  shuffle.textContent = "↻ 번호 바꾸기";
  shuffle.addEventListener("click", () => { pi = (pi + 1) % order.length; update(); });
  tools.append(shuffle);
  const ro = readouts();
  f.panel.append(pre.row, grid, tools, ro.row);
  f.onResize(draw);
  update();
}
