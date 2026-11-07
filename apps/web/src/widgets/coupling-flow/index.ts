// affine coupling 흐름 (data/widgets.toml: coupling-flow, 19.4장 "flow 구조").
// 본문 셀과 같은 고정 '신경망' s(u) = α·tanh(u)(본문 α = 0.6), t(u) = 1.2·sin(1.5u)로 coupling 층을 K개 쌓는다.
// 홀수 층은 첫 좌표를 고정하고 둘째 좌표를, 짝수 층은 둘째 좌표를 고정하고 첫 좌표를 바꾼다.
//   정방향  y_b = x_b·exp(s(x_a)) + t(x_a),  log|det| += s(x_a)
//   역변환  x_b = (y_b − t(y_a))·exp(−s(y_a))
// 왼쪽은 기저 공간 z ~ N(0, I), 오른쪽은 변환된 x와 변수변환 공식으로 계산한 밀도 p_X(x)다.
import { frame, param, readouts, el, C, rng, alpha, rgbOf } from "../frame";

type V2 = [number, number];
export function forward(z: V2, K: number, a: number): { x: V2; logdet: number } {
  let x0 = z[0], x1 = z[1], ld = 0;
  for (let k = 0; k < K; k++) {
    if (k % 2 === 0) { const s = a * Math.tanh(x0); x1 = x1 * Math.exp(s) + 1.2 * Math.sin(1.5 * x0); ld += s; }
    else { const s = a * Math.tanh(x1); x0 = x0 * Math.exp(s) + 1.2 * Math.sin(1.5 * x1); ld += s; }
  }
  return { x: [x0, x1], logdet: ld };
}
/** 역변환과 역방향 log|det dz/dx| = −Σ s */
export function inverse(x: V2, K: number, a: number): { z: V2; logdet: number } {
  let y0 = x[0], y1 = x[1], ld = 0;
  for (let k = K - 1; k >= 0; k--) {
    if (k % 2 === 0) { const s = a * Math.tanh(y0); y1 = (y1 - 1.2 * Math.sin(1.5 * y0)) * Math.exp(-s); ld -= s; }
    else { const s = a * Math.tanh(y1); y0 = (y0 - 1.2 * Math.sin(1.5 * y1)) * Math.exp(-s); ld -= s; }
  }
  return { z: [y0, y1], logdet: ld };
}
/** log p_X(x) = log N(z; 0, I) + log|det dz/dx| */
export function logProb(x: V2, K: number, a: number) {
  const { z, logdet } = inverse(x, K, a);
  return -0.5 * (z[0] * z[0] + z[1] * z[1]) - Math.log(2 * Math.PI) + logdet;
}
/** 격자 적분으로 잰 전체 질량. 상자는 반지름 R = 4 원(기저 질량의 1 − e^{−R²/2} ≈ 0.9997)의 상을 덮게 잡는다. */
export function totalMass(K: number, a: number, n = 800, R = 4) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < 720; i++) {
    const th = (2 * Math.PI * i) / 720;
    const { x } = forward([R * Math.cos(th), R * Math.sin(th)], K, a);
    x0 = Math.min(x0, x[0]); x1 = Math.max(x1, x[0]); y0 = Math.min(y0, x[1]); y1 = Math.max(y1, x[1]);
  }
  const hx = (x1 - x0) / n, hy = (y1 - y0) / n;
  let m = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m += Math.exp(logProb([x0 + (i + 0.5) * hx, y0 + (j + 0.5) * hy], K, a));
  return m * hx * hy;
}
/** 공식 log-det과 중앙차분 Jacobian의 log|det| 차이의 최댓값 */
export function jacobianGap(pts: V2[], K: number, a: number) {
  const h = 1e-5;
  let gap = 0;
  for (const z of pts) {
    const f = (d: V2) => forward([z[0] + d[0], z[1] + d[1]], K, a).x;
    const a1 = f([h, 0]), a2 = f([-h, 0]), b1 = f([0, h]), b2 = f([0, -h]);
    const J = [[(a1[0] - a2[0]) / (2 * h), (b1[0] - b2[0]) / (2 * h)], [(a1[1] - a2[1]) / (2 * h), (b1[1] - b2[1]) / (2 * h)]];
    const num = Math.log(Math.abs(J[0][0] * J[1][1] - J[0][1] * J[1][0]));
    gap = Math.max(gap, Math.abs(num - forward(z, K, a).logdet));
  }
  return gap;
}

export function mount(root: HTMLElement) {
  const narrow = root.clientWidth < 340;
  const f = frame(root, "affine coupling 흐름", "manipulate", narrow ? 1.0 : 0.52,
    "기저 공간의 점구름과 격자, 그리고 coupling 층을 지난 점구름·격자와 변수변환 공식으로 계산한 밀도");
  let K = 2, a = 0.6;
  const r = rng(11);
  const Z: V2[] = Array.from({ length: 300 }, () => [r.normal(), r.normal()]);
  let Xs: { x: V2; logdet: number }[] = [];
  const RES = 110;
  const off = document.createElement("canvas");
  off.width = off.height = RES;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(RES, RES);
  const ZB = 3.6, XB = 6.5; // 두 그림의 반폭

  function layout() {
    const W = f.width, H = f.height;
    if (W < 340) { const side = Math.min(W - 40, H / 2 - 30); return [{ x: (W - side) / 2, y: 20, s: side }, { x: (W - side) / 2, y: H / 2 + 14, s: side }]; }
    const side = Math.min(H - 44, W / 2 - (W < 560 ? 26 : 50));
    return [{ x: W / 4 - side / 2 + 4, y: 22, s: side }, { x: (3 * W) / 4 - side / 2 - 4, y: 22, s: side }];
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const [pz, px] = layout();
    const toP = (p: { x: number; y: number; s: number }, B: number) => (v: V2): V2 => [p.x + ((v[0] + B) / (2 * B)) * p.s, p.y + ((B - v[1]) / (2 * B)) * p.s];
    const PZ = toP(pz, ZB), PX = toP(px, XB);
    // 오른쪽 밀도 그림
    const [cr, cg, cb] = rgbOf(C.ink2);
    const dens = new Float64Array(RES * RES);
    let pmax = 0;
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
      const v = Math.exp(logProb([-XB + (2 * XB * (i + 0.5)) / RES, XB - (2 * XB * (j + 0.5)) / RES], K, a));
      dens[j * RES + i] = v; pmax = Math.max(pmax, v);
    }
    for (let k = 0; k < RES * RES; k++) {
      img.data[4 * k] = cr; img.data[4 * k + 1] = cg; img.data[4 * k + 2] = cb;
      img.data[4 * k + 3] = Math.round(235 * Math.sqrt(dens[k] / pmax));
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, px.x, px.y, px.s, px.s);
    // 격자선: z 공간의 수평·수직선과 그 상
    const lines: V2[][] = [];
    for (let c = -3; c <= 3; c++) {
      lines.push(Array.from({ length: 61 }, (_, i) => [c, -3 + (6 * i) / 60] as V2));
      lines.push(Array.from({ length: 61 }, (_, i) => [-3 + (6 * i) / 60, c] as V2));
    }
    for (const [p, P, B] of [[pz, PZ, ZB], [px, PX, XB]] as const) {
      ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.s, p.s); ctx.clip();
      ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
      for (const L of lines) {
        ctx.beginPath();
        L.forEach((z, i) => { const q = P === PZ ? PZ(z) : PX(forward(z, K, a).x); if (i) ctx.lineTo(...q); else ctx.moveTo(...q); });
        ctx.stroke();
      }
      // 점: z₁ < 0은 보조색, z₁ ≥ 0은 주색
      Z.forEach((z, i) => {
        const q = P === PZ ? PZ(z) : PX(Xs[i].x);
        ctx.fillStyle = z[0] < 0 ? alpha(C.ink2, 0.9) : C.edge;
        ctx.beginPath(); ctx.arc(q[0], q[1], 1.8, 0, 7); ctx.fill();
      });
      ctx.restore();
      ctx.strokeStyle = C.faint; ctx.strokeRect(p.x, p.y, p.s, p.s);
      ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillText(`±${B}`, p.x + p.s - 32, p.y + p.s - 5);
    }
    ctx.fillStyle = C.ink; ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("z ~ N(0, I)", pz.x, pz.y - 6);
    ctx.fillText(`x = f(z), 층 ${K}개 · 밀도 p_X(x)`, px.x, px.y - 6);
    if (W >= 560) { // 가운데 화살표
      const y = pz.y + pz.s / 2, x0 = pz.x + pz.s + 8, x1 = px.x - 8;
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.moveTo(x1 - 7, y - 5); ctx.lineTo(x1, y); ctx.lineTo(x1 - 7, y + 5); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  let timer = 0;
  function update() {
    Xs = Z.map((z) => forward(z, K, a));
    const meanLd = Xs.reduce((s, v) => s + v.logdet, 0) / Xs.length;
    let ld0 = Infinity, ld1 = -Infinity, err = 0;
    Xs.forEach((v, i) => { ld0 = Math.min(ld0, v.logdet); ld1 = Math.max(ld1, v.logdet); const zb = inverse(v.x, K, a).z; err = Math.max(err, Math.abs(zb[0] - Z[i][0]), Math.abs(zb[1] - Z[i][1])); });
    ro.set("ld", `log|det J| 평균 ${meanLd.toFixed(3)} · 범위 ${ld0.toFixed(2)}…${ld1.toFixed(2)}`, "hot");
    ro.set("inv", `역변환 오차 ${err.toExponential(0)} · log-det 검산 ${jacobianGap(Z.slice(0, 8), K, a).toExponential(0)}`, err < 1e-10 ? "ok" : "hot");
    draw();
    // 격자 적분(640,000점)은 조작이 멈춘 뒤에 한 번만 계산한다
    clearTimeout(timer);
    timer = window.setTimeout(() => { const m = totalMass(K, a); ro.set("mass", `∫p_X dx = ${m.toFixed(3)}`, Math.abs(m - 1) < 5e-3 ? "ok" : "hot"); }, 150);
  }

  const pk = param({ label: "K", min: 1, max: 4, step: 1, value: K, fmt: (v) => String(v), aria: "coupling 층 수", onInput: (v) => { K = v; update(); } });
  const pa = param({ label: "α", min: 0, max: 1, step: 0.05, value: a, fmt: (v) => v.toFixed(2), aria: "s(u) = α·tanh(u)의 α", onInput: (v) => { a = v; update(); } });
  const grid = el("div", "param-grid");
  grid.append(pk.row, pa.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
