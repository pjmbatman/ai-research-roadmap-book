// spread–skill과 rank histogram (data/widgets.toml: spread-skill).
// 27.2장의 Lorenz-96 앙상블: 분석 오차 0.5에 멤버 섭동 σ_p를 더해 20개 멤버를 참 동역학으로 10일(리드 40)까지 굴린다.
// σ_p를 분석 오차보다 작게 하면 과소분산(spread < RMSE, U자 rank histogram), 크게 하면 과대분산(돔 모양)이 된다.
// 왼쪽: 리드별 spread와 앙상블 평균 RMSE. 오른쪽: 고른 리드의 rank histogram(참값보다 작은 멤버 수 0–20).
import { frame, param, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { setup, ensemble, LEADS, M, ANALYSIS_SD } from "./model";

const SIGMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5];

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "spread–skill과 rank histogram", "manipulate", narrow ? 1.1 : 0.52, "리드 타임별 앙상블 spread와 앙상블 평균 RMSE 곡선, 고른 리드의 rank histogram");
  let si = SIGMAS.indexOf(0.2), lead = 4;
  const S = setup(0);
  const cache = new Map<number, ReturnType<typeof ensemble>>();
  const get = (s: number) => { let e = cache.get(s); if (!e) { e = ensemble(S, s); cache.set(s, e); } return e; };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const E = get(SIGMAS[si]);
    // ── 리드별 spread와 RMSE
    const A = narrow ? [34, 30, W - 10, H / 2 - 22] : [34, 30, W * 0.52 - 8, H - 24];
    {
      const [x0, y0, x1, y1] = A;
      const ymax = Math.max(4, ...E.rmse, ...E.spread) * 1.05;
      const X = (h: number) => x0 + (h / LEADS) * (x1 - x0), Y = (v: number) => y1 - (v / ymax) * (y1 - y0);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = C.mid;
      for (const v of [0, 1, 2, 3, 4]) { if (v > ymax) continue; ctx.strokeStyle = alpha(C.mid, 0.15); ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.stroke(); const s = String(v); ctx.fillText(s, x0 - 6 - ctx.measureText(s).width, Y(v) + 4); }
      for (const [h, s] of [[0, "0"], [20, "5일"], [40, "10일"]] as [number, string][]) ctx.fillText(s, Math.min(Math.max(X(h) - ctx.measureText(s).width / 2, x0), x1 - ctx.measureText(s).width), y1 + 14);
      const curve = (v: number[], col: string, w: number, dash: number[] = []) => {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
        ctx.beginPath(); v.forEach((y, h) => (h ? ctx.lineTo(X(h), Y(y)) : ctx.moveTo(X(h), Y(y)))); ctx.stroke(); ctx.setLineDash([]);
      };
      curve(E.rmse, C.ink, 1.8, [5, 4]);
      curve(E.spread, C.edge, 2.4);
      ctx.strokeStyle = C.mid; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(lead), y0); ctx.lineTo(X(lead), y1); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.edge; ctx.beginPath(); ctx.arc(X(lead), Y(E.spread[lead]), 4, 0, 7); ctx.fill();
      ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(lead), Y(E.rmse[lead]), 3.5, 0, 7); ctx.fill();
      let lx = x0;
      for (const [nm, col, dash] of [["평균 RMSE", C.ink, true], ["spread", C.edge, false]] as [string, string, boolean][]) {
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(lx, y0 - 14); ctx.lineTo(lx + 14, y0 - 14); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, y0 - 10);
        lx += 30 + ctx.measureText(nm).width;
      }
    }
    // ── rank histogram
    const B = narrow ? [34, H / 2 + 26, W - 10, H - 24] : [W * 0.52 + 30, 30, W - 10, H - 24];
    {
      const [x0, y0, x1, y1] = B;
      const r = E.ranks[lead];
      const ymax = Math.max(0.2, ...r) * 1.08;
      const bw = (x1 - x0) / (M + 1);
      const Y = (v: number) => y1 - (v / ymax) * (y1 - y0);
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      r.forEach((v, i) => { ctx.fillStyle = i === 0 || i === M ? C.edge : alpha(C.edge, 0.55); ctx.fillRect(x0 + i * bw + 1, Y(v), bw - 2, y1 - Y(v)); });
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(x0, Y(1 / (M + 1))); ctx.lineTo(x1, Y(1 / (M + 1))); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid;
      ctx.fillText("rank histogram", x0, y0 - 10);
      for (const i of [0, 10, 20]) { const s = String(i); ctx.fillText(s, x0 + (i + 0.5) * bw - ctx.measureText(s).width / 2, y1 + 14); }
      for (const v of [0, 0.1, 0.2]) { if (v > ymax) continue; const s = v.toFixed(1); ctx.fillText(s, x0 - 5 - ctx.measureText(s).width, Y(v) + 4); }
    }
  }

  let raf = 0;
  function update() {
    if (!raf) raf = requestAnimationFrame(() => {
      raf = 0;
      const E = get(SIGMAS[si]);
      const sp = E.spread[lead], rm = E.rmse[lead], ratio = sp / rm;
      const out = E.ranks[lead][0] + E.ranks[lead][M];
      const days = (lead * 6) / 24;
      ro.set("lead", `리드 ${lead} (${days % 1 ? days.toFixed(2) : days}일) · spread ${sp.toFixed(2)} · RMSE ${rm.toFixed(2)}`);
      ro.set("ratio", `spread/RMSE ${ratio.toFixed(2)} · ${ratio < 0.85 ? "과소분산" : ratio > 1.15 ? "과대분산" : "보정됨"}`, ratio < 0.85 || ratio > 1.15 ? "hot" : "ok");
      ro.set("out", `앙상블 밖 참값 ${(100 * out).toFixed(1)}% (보정 시 ${(200 / (M + 1)).toFixed(1)}%)`);
      pageState.set("spread-skill", { sigma: SIGMAS[si], lead, spread: sp, rmse: rm }, root);
      draw();
    });
  }

  const ps = param({ label: "σₚ", min: 0, max: SIGMAS.length - 1, step: 1, value: si, fmt: (v) => SIGMAS[v].toFixed(1), aria: `멤버 초기 섭동 표준편차 σ_p (분석 오차 ${ANALYSIS_SD})`, onInput: (v) => { si = v; update(); } });
  const pl = param({ label: "리드", min: 0, max: LEADS, step: 1, value: lead, fmt: (v) => `${v}`, aria: "리드 타임 (step, 1 step ≈ 6시간)", onInput: (v) => { lead = v; update(); } });
  const grid = el("div", "param-grid");
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  grid.append(ps.row, pl.row);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(() => { if (cache.size) draw(); });
  update();
}
