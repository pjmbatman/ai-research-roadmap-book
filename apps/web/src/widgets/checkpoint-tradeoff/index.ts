// gradient checkpointing (data/widgets.toml: checkpoint-tradeoff).
// 22.5장 "activation checkpointing"의 단순 모형: 같은 크기의 층 n개, 층 하나의 활성값 = 1 단위.
// 길이 k인 구간으로 나누면 forward는 구간 경계 ⌈n/k⌉개만 저장하고, backward에서 구간 하나를 다시 계산하는 동안 그 구간의 k개를 더 든다.
// 최대 저장량 = max_j (살아 있는 경계 j+1 + 구간 j의 길이) — n이 k로 나누어떨어지면 본문 표의 ⌈n/k⌉ + k와 같다(k = 1은 checkpointing 없음 = n).
// 재계산은 구간마다 (길이 − 1)층이고, backward = forward 2회로 보아 추가 계산 = (n − ⌈n/k⌉) / 3n (본문의 (1 − 1/k)/3).
// 위: 시간(forward 1층 = 1)에 따른 저장량. 아래: 모든 k의 (추가 계산, 최대 저장) 점.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";

export function plan(n: number, k: number) {
  const segs: number[] = [];
  for (let s = 0; s < n; s += k) segs.push(Math.min(k, n - s));
  const m = segs.length;
  // 시간–저장량 꺾은선: [t, 저장량]
  const pts: [number, number][] = [[0, 0]];
  let t = 0, peak = 0, recompute = 0;
  if (k === 1) {
    for (let i = 1; i <= n; i++) pts.push([++t, i]);
    for (let i = n - 1; i >= 0; i--) { t += 2; pts.push([t, i]); }
    peak = n;
  } else {
    for (let j = 0; j < m; j++) { pts.push([t, j + 1]); t += segs[j]; pts.push([t, j + 1]); }
    for (let j = m - 1; j >= 0; j--) {
      const base = j + 1, len = segs[j];
      // 재계산: (len − 1)층의 forward 동안 구간의 len개가 차오른다
      pts.push([t, base + 1]);
      t += len - 1; recompute += len - 1;
      pts.push([t, base + len]);
      peak = Math.max(peak, base + len);
      for (let i = len - 1; i >= 0; i--) { t += 2; pts.push([t, base + i]); }
      pts.push([t, j]);
    }
  }
  return { pts, peak, recompute, extra: recompute / (3 * n), total: t, segments: m };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "gradient checkpointing", "manipulate", narrow ? 1.0 : 0.58, "시간에 따른 활성값 저장량과, 구간 길이마다의 추가 계산–최대 저장량 점 그림");
  let n = 48, k = 6;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const L = 40, R = 14, top = 24, split = H * 0.5, bottom = H - 30;
    const cur = plan(n, k), none = plan(n, 1);
    const tMax = 4 * n, yMax = n + 2;
    const X = (t: number) => L + (t / tMax) * (W - L - R);
    const Y = (v: number) => split - 16 - (v / yMax) * (split - 16 - top);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textBaseline = "alphabetic";
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, split - 16); ctx.lineTo(W - R, split - 16); ctx.moveTo(L, top); ctx.lineTo(L, split - 16); ctx.stroke();
    ctx.fillStyle = C.mid;
    ctx.fillText(String(n), L - 6 - ctx.measureText(String(n)).width, Y(n) + 4);
    ctx.fillText("0", L - 12, Y(0) + 4);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("저장한 활성값(층) — 시간 →", L + 4, top - 8);
    // forward | backward 경계, 기본 학습의 끝(3n)
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.grid2;
    for (const t of [n, 3 * n]) { ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), split - 16); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    ctx.fillText("fwd", X(n / 2) - 10, split - 4);
    ctx.fillText("bwd", X(2 * n) - 10, split - 4);
    ctx.fillText("3n", X(3 * n) - 7, split - 4);
    // 기준: checkpointing 없음(회색 면)
    ctx.fillStyle = alpha(C.mid, 0.16);
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); none.pts.forEach(([t, v]) => ctx.lineTo(X(t), Y(v))); ctx.lineTo(X(none.total), Y(0)); ctx.closePath(); ctx.fill();
    // 현재 k
    ctx.fillStyle = alpha(C.ink2, 0.22);
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); cur.pts.forEach(([t, v]) => ctx.lineTo(X(t), Y(v))); ctx.lineTo(X(cur.total), Y(0)); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); cur.pts.forEach(([t, v], i) => (i ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)))); ctx.stroke();
    // 최대 저장량(코랄 가로선)
    ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(L, Y(cur.peak)); ctx.lineTo(W - R, Y(cur.peak)); ctx.stroke();
    ctx.fillStyle = C.edge; ctx.font = "600 12px Pretendard Variable, sans-serif";
    const pl = `최대 ${cur.peak}`;
    ctx.fillText(pl, W - R - ctx.measureText(pl).width, Y(cur.peak) - 5);
    // 재계산으로 늘어난 시간
    if (cur.total > 3 * n) {
      ctx.fillStyle = alpha(C.edge, 0.18);
      ctx.fillRect(X(3 * n), top, X(cur.total) - X(3 * n), split - 16 - top);
    }

    // 아래: 모든 k의 (추가 계산 %, 최대 저장)
    const ks = Array.from({ length: n }, (_, i) => i + 1).map((kk) => ({ k: kk, ...plan(n, kk) }));
    const top2 = split + 22, L2 = 40;
    const X2 = (e: number) => L2 + (e / 0.36) * (W - L2 - R);
    const Y2 = (v: number) => bottom - (v / yMax) * (bottom - top2);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L2, bottom); ctx.lineTo(W - R, bottom); ctx.moveTo(L2, top2); ctx.lineTo(L2, bottom); ctx.stroke();
    ctx.fillStyle = C.mid; ctx.font = "11px JetBrains Mono, monospace";
    for (const e of [0, 0.1, 0.2, 0.3]) { const s = `${Math.round(e * 100)}%`; ctx.fillText(s, X2(e) - ctx.measureText(s).width / 2, bottom + 15); }
    ctx.fillText(String(n), L2 - 6 - ctx.measureText(String(n)).width, Y2(n) + 4);
    ctx.fillText("0", L2 - 12, Y2(0) + 4);
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillText("k마다: 추가 계산 → 최대 저장 ↑", L2 + 4, top2 - 8);
    // 식 n/k + k의 최소 2√n
    const smin = 2 * Math.sqrt(n);
    ctx.setLineDash([3, 4]); ctx.strokeStyle = C.good; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(L2, Y2(smin)); ctx.lineTo(W - R, Y2(smin)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.good; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText("2√n", L2 + 4, Y2(smin) + 13);
    ctx.strokeStyle = C.grid2; ctx.lineWidth = 1;
    ctx.beginPath(); ks.forEach((p, i) => (i ? ctx.lineTo(X2(p.extra), Y2(p.peak)) : ctx.moveTo(X2(p.extra), Y2(p.peak)))); ctx.stroke();
    for (const p of ks) {
      ctx.fillStyle = C.ink2;
      ctx.beginPath(); ctx.arc(X2(p.extra), Y2(p.peak), 2.6, 0, 7); ctx.fill();
    }
    const c = ks[k - 1];
    ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X2(c.extra), Y2(c.peak), 6, 0, 7); ctx.fill();
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    const lab = `k = ${k}`, lx = X2(c.extra) + 9;
    ctx.fillText(lab, lx + ctx.measureText(lab).width > W - R ? X2(c.extra) - 9 - ctx.measureText(lab).width : lx, Y2(c.peak) - 8);
  }

  function update() {
    const p = plan(n, k);
    ro.set("peak", `최대 저장 ${p.peak}층 (없으면 ${n})`, "hot");
    ro.set("extra", `추가 계산 +${Math.round(p.extra * 100)}%`);
    ro.set("formula", `식 n/k + k = ${(k === 1 ? n : n / k + k).toFixed(1)} · 최소 2√n = ${(2 * Math.sqrt(n)).toFixed(1)}`);
    ro.set("seg", `구간 ${p.segments}개`);
    pre.select(k === 1 ? "none" : k === n ? "all" : k === best() ? "sqrt" : null);
    draw();
  }

  type P = "none" | "sqrt" | "all";
  const best = () => { let bk = 1, bp = Infinity; for (let kk = 2; kk <= n; kk++) { const q = plan(n, kk).peak; if (q < bp) { bp = q; bk = kk; } } return bk; };
  const pick = (key: P) => { k = key === "none" ? 1 : key === "all" ? n : best(); pk.set(k); update(); };
  const pre = presets<P>([{ key: "none", label: "없음 (k = 1)" }, { key: "sqrt", label: "k ≈ √n" }, { key: "all", label: "k = n" }], pick);
  pre.row.classList.add("segmented");
  const pn = param({ label: "층 n", min: 4, max: 96, step: 1, value: n, fmt: (v) => String(v), aria: "같은 크기 층의 수 n", onInput: (v) => {
    n = v; k = Math.min(k, n); pk.input.max = String(n); pk.set(k); update();
  } });
  const pk = param({ label: "구간 k", min: 1, max: n, step: 1, value: k, fmt: (v) => String(v), aria: "checkpoint 구간 길이 k", onInput: (v) => { k = v; update(); } });
  pn.row.style.gridTemplateColumns = pk.row.style.gridTemplateColumns = "3.6em minmax(0, 1fr) 2.4em";
  const grid = el("div", "param-grid");
  grid.append(pn.row, pk.row);
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
