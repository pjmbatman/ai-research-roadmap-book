// 빔 설계 Pareto (data/widgets.toml: beam-pareto).
// 26.4장 셀 1과 같은 외팔보: 왼쪽은 설계 공간(폭 b × 높이 h)과 응력 제약 경계, 오른쪽은 목적 공간(질량 대 끝단 처짐).
// 폭·높이 슬라이더로 설계 하나를 고르고, 허용 응력을 바꾸면 제약 경계와 Pareto front가 함께 움직인다.
// Pareto 최적 설계는 모두 폭 하한(1 cm)이나 높이 상한(10 cm) 위에 놓인다.
import { frame, param, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { BS, HS, F_N, L_M, evalDesign, pareto, dominator, type Design } from "./model";

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "빔 설계 Pareto", "manipulate", narrow ? 1.1 : 0.56, "외팔보의 설계 공간(폭·높이)과 응력 제약 경계, 질량 대 처짐 목적 공간의 Pareto front와 고른 설계");
  let bi = 0, hi = 43, sig = 160; // 폭 1.0 cm, 높이 6.3 cm: 처짐 8 mm 이하에서 가장 가벼운 설계(본문 셀 1)
  let P = pareto(sig * 1e6);
  let frontSorted: Design[] = [];
  const refront = () => { P = pareto(sig * 1e6); frontSorted = [...P.front].sort((a, b) => a.mass - b.mass); };
  refront();

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const cur = evalDesign(BS[bi], HS[hi]);
    const isFront = P.front.has(P.ok.find((d) => d.b === cur.b && d.h === cur.h) as Design);
    const okCur = cur.sigma <= sig * 1e6;
    // 두 패널의 상자
    const A = narrow ? [34, 22, W - 10, H / 2 - 22] : [34, 22, W / 2 - 14, H - 24];
    const B = narrow ? [40, H / 2 + 22, W - 10, H - 24] : [W / 2 + 32, 22, W - 10, H - 24];
    // ── 설계 공간
    {
      const [x0, y0, x1, y1] = A;
      const X = (b: number) => x0 + ((b - 0.01) / 0.04) * (x1 - x0), Y = (h: number) => y1 - ((h - 0.02) / 0.08) * (y1 - y0);
      // 응력 제약 위반 영역: h < sqrt(6 F L / (b σ))
      const hmin = (b: number) => Math.sqrt((6 * F_N * L_M) / (b * sig * 1e6));
      ctx.fillStyle = alpha(C.mid, 0.16);
      ctx.beginPath(); ctx.moveTo(X(0.01), Y(0.02));
      for (let b = 0.01; b <= 0.05 + 1e-9; b += 0.0005) ctx.lineTo(X(b), Y(Math.min(Math.max(hmin(b), 0.02), 0.1)));
      ctx.lineTo(X(0.05), Y(0.02)); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
      ctx.beginPath(); let started = false;
      for (let b = 0.01; b <= 0.05 + 1e-9; b += 0.0005) { const h = hmin(b); if (h > 0.1 || h < 0.02) { started = false; continue; } started ? ctx.lineTo(X(b), Y(h)) : ctx.moveTo(X(b), Y(h)); started = true; }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      // Pareto 설계
      ctx.fillStyle = C.edge;
      P.front.forEach((d) => { ctx.beginPath(); ctx.arc(X(d.b), Y(d.h), 2.3, 0, 7); ctx.fill(); });
      // 고른 설계
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(X(cur.b), y0); ctx.lineTo(X(cur.b), y1); ctx.moveTo(x0, Y(cur.h)); ctx.lineTo(x1, Y(cur.h)); ctx.stroke(); ctx.setLineDash([]);
      ctx.lineWidth = 2.2; ctx.strokeStyle = okCur ? C.ink : C.mid;
      ctx.beginPath(); ctx.arc(X(cur.b), Y(cur.h), 6, 0, 7); ctx.stroke();
      ctx.fillStyle = C.mid;
      for (const b of [1, 2, 3]) { const s = String(b); ctx.fillText(s, X(b / 100) - ctx.measureText(s).width / 2, y1 + 13); }
      for (const h of [2, 6, 10]) { const s = String(h); ctx.fillText(s, x0 - 5 - ctx.measureText(s).width, Y(h / 100) + 4); }
      ctx.fillText("높이 h [cm]", x0, y0 - 8);
      const s = "폭 [cm]"; ctx.fillText(s, x1 - ctx.measureText(s).width, y1 + 13);
      ctx.fillStyle = C.mid;
      const t = `σ > ${sig} MPa`; ctx.fillText(t, x1 - 4 - ctx.measureText(t).width, y1 - 6);
    }
    // ── 목적 공간: 질량 대 처짐(로그)
    {
      const [x0, y0, x1, y1] = B;
      const M1 = 40, lo = Math.log10(0.3), hi2 = Math.log10(300);
      const X = (m: number) => x0 + (m / M1) * (x1 - x0), Y = (d: number) => y1 - ((Math.log10(d) - lo) / (hi2 - lo)) * (y1 - y0);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = C.mid;
      for (const d of [1, 10, 100]) {
        ctx.strokeStyle = alpha(C.mid, 0.15); ctx.beginPath(); ctx.moveTo(x0, Y(d)); ctx.lineTo(x1, Y(d)); ctx.stroke();
        const s = String(d); ctx.fillText(s, x0 - 5 - ctx.measureText(s).width, Y(d) + 4);
      }
      for (const m of [0, 10, 20]) { const s = String(m); ctx.fillText(s, Math.min(X(m) - ctx.measureText(s).width / 2, x1 - ctx.measureText(s).width), y1 + 13); }
      ctx.fillText("처짐 δ [mm]", x0, y0 - 8);
      const s = "질량 [kg]"; ctx.fillText(s, x1 - ctx.measureText(s).width, y1 + 13);
      ctx.fillStyle = alpha(C.ink2, 0.35);
      for (const d of P.ok) ctx.fillRect(X(d.mass) - 1, Y(d.defl) - 1, 2, 2);
      ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
      ctx.beginPath(); frontSorted.forEach((d, i) => (i ? ctx.lineTo(X(d.mass), Y(d.defl)) : ctx.moveTo(X(d.mass), Y(d.defl)))); ctx.stroke();
      ctx.fillStyle = C.edge;
      frontSorted.forEach((d) => { ctx.beginPath(); ctx.arc(X(d.mass), Y(d.defl), 2.3, 0, 7); ctx.fill(); });
      const dy = Math.min(Math.max(Y(cur.defl), y0), y1);
      ctx.lineWidth = 2.2; ctx.strokeStyle = okCur ? C.ink : C.mid;
      if (!okCur) ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.arc(X(cur.mass), dy, 6, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      if (isFront) { ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X(cur.mass), dy, 3.5, 0, 7); ctx.fill(); }
    }
  }

  let raf = 0;
  function update(front = false) {
    if (front) refront();
    const cur = evalDesign(BS[bi], HS[hi]);
    const okCur = cur.sigma <= sig * 1e6;
    const match = P.ok.find((d) => d.b === cur.b && d.h === cur.h);
    const onFront = !!match && P.front.has(match);
    ro.set("obj", `질량 ${cur.mass.toFixed(2)} kg · 처짐 ${cur.defl < 10 ? cur.defl.toFixed(2) : cur.defl.toFixed(1)} mm`);
    ro.set("sig", `응력 ${(cur.sigma / 1e6).toFixed(0)} MPa ${okCur ? "≤" : ">"} ${sig}`, okCur ? "ok" : "hot");
    let st = "제약 위반: 설계 불가";
    if (okCur) {
      const q = onFront ? null : dominator(cur, P.ok);
      st = onFront ? "Pareto 최적" : q ? `지배됨: ${(q.b * 100).toFixed(1)}×${(q.h * 100).toFixed(1)} cm가 ${q.mass.toFixed(2)} kg · ${q.defl < 10 ? q.defl.toFixed(2) : q.defl.toFixed(1)} mm` : "Pareto 최적";
    }
    ro.set("dom", st, onFront ? "hot" : "");
    ro.set("count", `제약 만족 ${P.ok.length}개 중 Pareto ${P.front.size}개`);
    pageState.set("beam-pareto", { b_cm: cur.b * 100, h_cm: cur.h * 100, sigma_allow_mpa: sig, pareto: onFront }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pb = param({ label: "폭", min: 0, max: BS.length - 1, step: 1, value: bi, fmt: (v) => `${(BS[v] * 100).toFixed(1)} cm`, aria: "폭 b (1–5 cm)", onInput: (v) => { bi = v; update(); } });
  const ph = param({ label: "높이", min: 0, max: HS.length - 1, step: 1, value: hi, fmt: (v) => `${(HS[v] * 100).toFixed(1)} cm`, aria: "높이 h (2–10 cm)", onInput: (v) => { hi = v; update(); } });
  const ps = param({ label: "σ", min: 60, max: 300, step: 10, value: sig, fmt: (v) => `${v} MPa`, aria: "허용 응력 (MPa)", onInput: (v) => { sig = v; update(true); } });
  for (const p of [pb, ph, ps]) p.row.style.gridTemplateColumns = "2.4em minmax(0, 1fr) 4.6em";
  const grid = el("div", "param-grid");
  grid.append(pb.row, ph.row, ps.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
