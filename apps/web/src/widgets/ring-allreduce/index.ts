// ring all-reduce (data/widgets.toml: ring-allreduce).
// 22.6장 "통신 병목": GPU W개를 고리로 잇고 gradient 배열을 W개 청크로 나눈다.
// reduce-scatter W−1단계: 단계 s에서 GPU i가 청크 (i − s) mod W를 이웃 i+1에 보내고, 받은 쪽은 자기 값에 더한다.
// all-gather W−1단계: 단계 s에서 GPU i가 완성된 청크 (i + 1 − s) mod W를 이웃에 보내고, 받은 쪽은 덮어쓴다.
// 칸 하나 = (GPU, 청크), 칸 안의 작은 막대 W개 = 그 청크에 더해진 GPU들의 몫. 단계마다 GPU 하나가 S/W를 보내므로 끝까지 2(W−1)/W·S.
// S는 본문 예의 3.5억 파라미터 fp32 gradient 1.40 GB.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

const S_GB = 350e6 * 4 / 1e9;

/** t단계 뒤의 상태: cells[i][c] = GPU i의 청크 c에 더해진 GPU 집합(bitmask). sent = t번째 단계에서 보낸 청크(i가 보냄). */
export function simulate(W: number, t: number) {
  const cells = Array.from({ length: W }, (_, i) => Array.from({ length: W }, () => 1 << i));
  let sent: number[] = [];
  for (let step = 0; step < t; step++) {
    const rs = step < W - 1, s = rs ? step : step - (W - 1);
    const snap = cells.map((r) => [...r]);
    sent = [];
    for (let i = 0; i < W; i++) {
      const c = (((rs ? i - s : i + 1 - s) % W) + W) % W, j = (i + 1) % W;
      cells[j][c] = rs ? snap[j][c] | snap[i][c] : snap[i][c];
      sent.push(c);
    }
  }
  return { cells, sent, sentGB: (t * S_GB) / W };
}
const popcount = (x: number) => { let c = 0; while (x) { c += x & 1; x >>= 1; } return c; };

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "ring all-reduce", "step", narrow ? 0.98 : 0.62, "고리로 이은 GPU들이 gradient 청크를 이웃에게 넘기며 더하고(reduce-scatter) 완성된 청크를 돌리는(all-gather) 과정");
  let W = 4, t = 0;

  function draw() {
    const { ctx, width: Wd, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, Wd, H);
    const st = simulate(W, t);
    const stripH = 40, M = 10;
    // 고리를 경주로 모양으로 편다: 위 줄은 GPU 0..T−1(왼→오), 아래 줄은 나머지(오→왼). 화살표 i → i+1이 한 바퀴를 돈다
    const T = Math.ceil(W / 2), gapX = Math.min(34, Wd * 0.07), gapY = 34;
    const bw = Math.min(150, (Wd - 2 * M - (T - 1) * gapX) / T);
    const bh = Math.min(130, (H - stripH - 2 * M - gapY) / 2);
    const x0s = (Wd - (T * bw + (T - 1) * gapX)) / 2;
    const box = (i: number) => {
      const top = i < T, col = top ? i : T - 1 - (i - T);
      return { x: x0s + col * (bw + gapX), y: M + (top ? 0 : bh + gapY) };
    };
    const ctr = (i: number) => { const b = box(i); return [b.x + bw / 2, b.y + bh / 2]; };
    // 상자 가장자리까지 자른 화살표
    const clip = (cx: number, cy: number, dx: number, dy: number) => {
      const sx = dx ? (bw / 2 + 3) / Math.abs(dx) : Infinity, sy = dy ? (bh / 2 + 3) / Math.abs(dy) : Infinity, k = Math.min(sx, sy);
      return [cx + dx * k, cy + dy * k];
    };
    for (let i = 0; i < W; i++) {
      const j = (i + 1) % W, [ax, ay] = ctr(i), [bx, by] = ctr(j);
      let dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy); dx /= len; dy /= len;
      const [px, py] = W === 2 ? [i === 0 ? 16 : -16, 0] : [0, 0];   // W = 2는 위아래 두 화살표가 겹치지 않게 비켜 놓는다
      const [sx, sy] = clip(ax, ay, dx, dy), [ex, ey] = clip(bx, by, -dx, -dy);
      const active = t > 0;
      ctx.strokeStyle = active ? C.edge : C.faint; ctx.lineWidth = active ? 2.4 : 1.5;
      ctx.beginPath(); ctx.moveTo(sx + px, sy + py); ctx.lineTo(ex + px, ey + py); ctx.stroke();
      ctx.fillStyle = active ? C.edge : C.faint; ctx.beginPath();
      ctx.moveTo(ex + px, ey + py); ctx.lineTo(ex + px - 8 * dx - 4.5 * dy, ey + py - 8 * dy + 4.5 * dx); ctx.lineTo(ex + px - 8 * dx + 4.5 * dy, ey + py - 8 * dy - 4.5 * dx); ctx.closePath(); ctx.fill();
      if (active) {
        ctx.font = "700 11px JetBrains Mono, monospace";
        const lab = `c${st.sent[i]}`, lw = ctx.measureText(lab).width;
        const mx = (sx + ex) / 2 + px, my = (sy + ey) / 2 + py;
        // 세로 화살표는 경주로 바깥쪽에, 가로 화살표는 선 위에 청크 번호를 쓴다
        const side = Math.sign(mx - Wd / 2) || Math.sign(px) || 1;
        const nx = Math.abs(dy) > 0.5 ? side * (lw / 2 + 7) : 0, ny = Math.abs(dy) > 0.5 ? 4 : -7;
        ctx.fillStyle = C.bg; ctx.fillRect(mx + nx - lw / 2 - 2, my + ny - 9, lw + 4, 12);
        ctx.fillStyle = C.edge; ctx.fillText(lab, mx + nx - lw / 2, my + ny + 1);
      }
    }
    // GPU 상자: 청크 W칸, 칸마다 더해진 GPU 몫의 막대 W개(자기 몫 = 회색, 받은 몫 = 진한 파랑, 모두 더해짐 = 초록)
    const pad = 7, cw = (bw - 2 * pad) / W, gy0 = 18, gh = bh - gy0 - 16, slot = gh / W;
    for (let i = 0; i < W; i++) {
      const { x: x0, y: y0 } = box(i);
      ctx.fillStyle = C.bg; ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(x0, y0, bw, bh, 8); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.font = "700 11px Pretendard Variable, sans-serif";
      ctx.fillText(`GPU ${i}`, x0 + pad, y0 + 13);
      const gy = y0 + gy0;
      for (let c = 0; c < W; c++) {
        const m = st.cells[i][c], full = popcount(m) === W, gx = x0 + pad + c * cw;
        ctx.fillStyle = alpha(C.mid, 0.1); ctx.fillRect(gx + 1, gy, cw - 2, gh);
        for (let g = 0; g < W; g++) {
          if (!(m & (1 << g))) continue;
          ctx.fillStyle = full ? C.good : g === i ? C.mid : C.ink2;
          ctx.fillRect(gx + 2, gy + gh - (g + 1) * slot + 1, cw - 4, Math.max(1.5, slot - 2));
        }
        const recv = t > 0 && st.sent[(i - 1 + W) % W] === c, send = t > 0 && st.sent[i] === c;
        if (recv || send) { ctx.strokeStyle = C.edge; ctx.lineWidth = recv ? 2.2 : 1.2; ctx.setLineDash(recv ? [] : [3, 2]); ctx.strokeRect(gx + 0.5, gy - 1, cw - 1, gh + 2); ctx.setLineDash([]); }
        if (cw >= 10) { ctx.fillStyle = C.mid; ctx.font = "10px JetBrains Mono, monospace"; ctx.fillText(String(c), gx + cw / 2 - 3, gy + gh + 11); }
      }
    }
    // 아래 띠: 단계 진행
    const steps = 2 * (W - 1), L = 12, R = 12, sy = H - stripH + 20;
    const sw = (Wd - L - R) / steps;
    for (let s = 0; s < steps; s++) {
      const rs = s < W - 1;
      ctx.fillStyle = s < t ? (rs ? C.ink2 : C.good) : alpha(C.mid, 0.15);
      ctx.fillRect(L + s * sw + 1, sy, sw - 2, 10);
      if (s === t - 1) { ctx.strokeStyle = C.edge; ctx.lineWidth = 2; ctx.strokeRect(L + s * sw, sy - 1, sw, 12); }
    }
    ctx.fillStyle = C.mid; ctx.font = "600 11px Pretendard Variable, sans-serif";
    ctx.fillText("reduce-scatter", L, sy - 5);
    const ag = "all-gather";
    ctx.fillText(ag, Math.min(L + (W - 1) * sw + 2, Wd - R - ctx.measureText(ag).width), sy - 5);
  }

  function update() {
    const st = simulate(W, t), steps = 2 * (W - 1);
    const phase = t === 0 ? "시작: 각자 자기 gradient" : t <= W - 1 ? `reduce-scatter ${t}/${W - 1}` : `all-gather ${t - (W - 1)}/${W - 1}`;
    ro.set("phase", `단계 ${t}/${steps} · ${phase}`, "hot");
    let full = 0; st.cells.forEach((r) => r.forEach((m) => { if (popcount(m) === W) full++; }));
    ro.set("full", `완성된 칸 ${full}/${W * W}`, full === W * W ? "ok" : "");
    ro.set("sent", `GPU당 보낸 양 ${st.sentGB.toFixed(2)} / 2(W−1)/W·S = ${(2 * (W - 1) / W * S_GB).toFixed(2)} GB`);
    pt.input.max = String(steps);
    pre.select(t === 0 ? "start" : t === W - 1 ? "rs" : t === steps ? "end" : null);
    draw();
  }

  type P = "start" | "rs" | "end";
  const pre = presets<P>([{ key: "start", label: "처음" }, { key: "rs", label: "reduce-scatter 끝" }, { key: "end", label: "끝" }], (k) => {
    t = k === "start" ? 0 : k === "rs" ? W - 1 : 2 * (W - 1); pt.set(t); update();
  }, "start");
  pre.row.classList.add("segmented");
  const pw = param({ label: "GPU W", min: 2, max: 8, step: 1, value: W, fmt: (v) => String(v), aria: "GPU 수 W", onInput: (v) => { W = v; t = Math.min(t, 2 * (W - 1)); pt.set(t); update(); } });
  const pt = param({ label: "단계", min: 0, max: 2 * (W - 1), step: 1, value: t, fmt: (v) => String(v), aria: "진행 단계", onInput: (v) => { t = v; update(); } });
  pw.row.style.gridTemplateColumns = pt.row.style.gridTemplateColumns = "4.2em minmax(0, 1fr) 2em";
  const grid = el("div", "param-grid");
  grid.append(pw.row, pt.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  // 그림을 누르면 다음 단계(끝에서는 처음으로)
  f.canvas.style.cursor = "pointer";
  f.canvas.addEventListener("click", () => { t = t >= 2 * (W - 1) ? 0 : t + 1; pt.set(t); update(); });
  f.onResize(draw);
  update();
}
