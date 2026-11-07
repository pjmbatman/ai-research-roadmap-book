// 위도 가중 평균 (data/widgets.toml: area-weighting).
// 27.1장 셀 1과 같은 기온장 T(φ) = 300 − 45 sin²φ [K]를 규칙 위경도 격자(위도·경도 간격 Δ)에 놓고 평균한다.
// 칸 면적 A = 2R²Δλ cos φ sin(Δφ/2)는 cos φ에 비례하므로, 단순 평균은 면적이 작은 극 쪽 칸에 적도 칸과 같은 표를 준다.
// 왼쪽: 지구본 위의 격자와 적도·극 쪽 칸 하나씩. 오른쪽: 위도 띠별 가중치(단순 = 1, 면적 = cos φ / 평균 cos φ)와 기온, 두 평균.
import { frame, param, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

const RES = [1, 2, 2.5, 3, 5, 6, 9, 10, 15, 18, 30];
const R_KM = 6371;
const rad = (d: number) => (d * Math.PI) / 180;
export const temp = (latDeg: number) => 300 - 45 * Math.sin(rad(latDeg)) ** 2;

/** 간격 Δ°의 격자에서 단순 평균, cos 가중 평균, 극 쪽(|φ| > 60°) 칸의 개수·면적 비율 */
export function stats(d: number) {
  const lats = Array.from({ length: Math.round(180 / d) }, (_, i) => -90 + d / 2 + i * d);
  const w = lats.map((p) => Math.cos(rad(p)));
  const W = w.reduce((a, b) => a + b, 0);
  const simple = lats.reduce((s, p) => s + temp(p), 0) / lats.length; // 경도 방향은 모두 같으므로 위도 띠 평균과 같다
  const weighted = lats.reduce((s, p, i) => s + w[i] * temp(p), 0) / W;
  const polar = lats.map((p) => Math.abs(p) > 60);
  return {
    lats, w, simple, weighted,
    polarCount: polar.filter(Boolean).length / lats.length,
    polarArea: w.reduce((s, v, i) => s + (polar[i] ? v : 0), 0) / W,
    area: (p: number) => 2 * R_KM ** 2 * rad(d) * Math.cos(rad(p)) * Math.sin(rad(d) / 2), // km²
  };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "위도 가중 평균", "manipulate", narrow ? 1.15 : 0.5, "위경도 격자를 그린 지구본과 위도 띠별 평균 가중치, 기온 곡선, 단순 평균과 면적 가중 평균");
  let ri = RES.indexOf(5), weighted = false;

  function globe(cx: number, cy: number, r: number, d: number, S: ReturnType<typeof stats>) {
    const { ctx } = f;
    const e = rad(22), lon0 = rad(-20);
    const proj = (latD: number, lonD: number) => {
      const p = rad(latD), l = rad(lonD) - lon0;
      return { x: cx + r * Math.cos(p) * Math.sin(l), y: cy - r * (Math.sin(p) * Math.cos(e) - Math.cos(p) * Math.cos(l) * Math.sin(e)), z: Math.sin(p) * Math.sin(e) + Math.cos(p) * Math.cos(l) * Math.cos(e) };
    };
    const path = (pts: { x: number; y: number; z: number }[]) => { let on = false; for (const q of pts) { if (q.z < 0) { on = false; continue; } on ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); on = true; } };
    ctx.fillStyle = alpha(C.ink2, 0.08); ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.fill();
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.stroke();
    // 격자선: 1° 격자는 너무 촘촘하므로 그리는 간격만 5° 이상으로 묶는다(칸 계산은 Δ 그대로)
    const step = d * Math.max(1, Math.ceil(5 / d));
    ctx.strokeStyle = alpha(C.mid, 0.45); ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let p = -90; p <= 90 + 1e-9; p += step) path(Array.from({ length: 145 }, (_, i) => proj(p, -180 + i * 2.5)));
    for (let l = -180; l < 180; l += step) path(Array.from({ length: 73 }, (_, i) => proj(-90 + i * 2.5, l)));
    ctx.stroke();
    // 경도 30° 폭(또는 한 칸)의 부채꼴 안 칸 전부: 극으로 갈수록 칸이 가늘어진다(|φ| > 60°는 진하게)
    const cell = (p0: number, l0: number, col: string) => {
      const pts = [...Array.from({ length: 11 }, (_, i) => proj(p0, l0 + (d * i) / 10)), ...Array.from({ length: 11 }, (_, i) => proj(p0 + d, l0 + d - (d * i) / 10))];
      if (pts.some((q) => q.z < 0)) return;
      ctx.fillStyle = col; ctx.beginPath(); pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = C.bg; ctx.lineWidth = d >= 5 ? 1 : 0.5; ctx.stroke();
    };
    const width = Math.max(d, 30), ncol = Math.round(width / d);
    for (let c = 0; c < ncol; c++) S.lats.forEach((p) => cell(p - d / 2, -20 - width / 2 + c * d, Math.abs(p) > 60 ? C.edge : alpha(C.edge, 0.45)));
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const d = RES[ri], S = stats(d);
    // 지구본
    const gr = narrow ? Math.min(W * 0.32, H * 0.2) : Math.min(W * 0.18, H * 0.42);
    const gcx = narrow ? W / 2 : 14 + gr, gcy = narrow ? 12 + gr : H / 2;
    globe(gcx, gcy, gr, d, S);
    // 위도 띠별 가중치와 기온
    const x0 = narrow ? 34 : gcx + gr + 40, x1 = W - 38, y0 = narrow ? gcy + gr + 34 : 30, y1 = H - 24;
    const X = (p: number) => x0 + ((p + 90) / 180) * (x1 - x0);
    const Y = (v: number) => y1 - (v / 1.7) * (y1 - y0);
    const TY = (t: number) => y1 - ((t - 250) / 55) * (y1 - y0);
    const meanW = S.w.reduce((a, b) => a + b, 0) / S.w.length;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const bw = ((x1 - x0) * d) / 180;
    S.lats.forEach((p, i) => {
      const area = S.w[i] / meanW, used = weighted ? area : 1;
      ctx.fillStyle = Math.abs(p) > 60 ? alpha(C.edge, 0.85) : alpha(C.edge, 0.45);
      ctx.fillRect(X(p - d / 2) + (bw > 4 ? 0.5 : 0), Y(used), Math.max(bw - (bw > 4 ? 1 : 0), 0.6), y1 - Y(used));
    });
    // 반대 쪽 가중치(겹쳐 그린 윤곽)
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    if (weighted) { ctx.moveTo(x0, Y(1)); ctx.lineTo(x1, Y(1)); }
    else S.lats.forEach((p, i) => { const y = Y(S.w[i] / meanW); i ? ctx.lineTo(X(p - d / 2), y) : ctx.moveTo(X(p - d / 2), y); ctx.lineTo(X(p + d / 2), y); });
    ctx.stroke(); ctx.setLineDash([]);
    // 기온 곡선과 평균
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
    ctx.beginPath(); for (let p = -90; p <= 90; p += 2) (p === -90 ? ctx.moveTo(X(p), TY(temp(p))) : ctx.lineTo(X(p), TY(temp(p)))); ctx.stroke();
    for (const [v, on] of [[S.simple, !weighted], [S.weighted, weighted]] as [number, boolean][]) {
      ctx.strokeStyle = on ? C.ink : alpha(C.mid, 0.6); ctx.lineWidth = on ? 1.8 : 1; ctx.setLineDash(on ? [] : [2, 3]);
      ctx.beginPath(); ctx.moveTo(x0, TY(v)); ctx.lineTo(x1, TY(v)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = on ? C.ink : C.mid; const s = v.toFixed(1); ctx.fillText(s, x1 + 4, TY(v) + (v === S.simple ? 9 : -2));
    }
    ctx.fillStyle = C.mid;
    for (const p of [-60, 0, 60]) { const s = `${p}°`; ctx.fillText(s, Math.min(Math.max(X(p) - ctx.measureText(s).width / 2, x0 - 6), x1 - ctx.measureText(s).width + 6), y1 + 14); }
    for (const v of [0, 1]) { const s = String(v); ctx.fillText(s, x0 - 6 - ctx.measureText(s).width, Y(v) + 4); }
    ctx.fillText("가중치", x0, y0 - 8);
    const tK = "T [K]"; ctx.fillText(tK, x1 + 4 - ctx.measureText(tK).width + 30, y0 - 8);
  }

  let raf = 0;
  function update() {
    const d = RES[ri], S = stats(d);
    const v = weighted ? S.weighted : S.simple;
    ro.set("mean", `${weighted ? "cos φ 가중" : "단순"} 평균 ${v.toFixed(2)} K (해석해 285 K)`, Math.abs(v - 285) < 0.5 ? "ok" : "hot");
    const dd = S.simple - S.weighted;
    ro.set("diff", `단순 − 가중 = ${dd < 0 ? "−" : "+"}${Math.abs(dd).toFixed(2)} K`);
    ro.set("polar", `|φ| > 60°: 칸 ${(100 * S.polarCount).toFixed(1)}% · 면적 ${(100 * S.polarArea).toFixed(1)}%`);
    const eq = S.area(S.lats[Math.floor(S.lats.length / 2)]), po = S.area(S.lats[S.lats.length - 1]);
    const km = (a: number) => Math.round(a).toLocaleString("en-US");
    ro.set("area", `칸 면적 적도 ${km(eq)} · 극 ${km(po)} km²`);
    pageState.set("area-weighting", { resolution_deg: d, weighted, mean_k: v }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pr = param({ label: "Δ", min: 0, max: RES.length - 1, step: 1, value: ri, fmt: (v) => `${RES[v]}°`, aria: "격자 간격 Δ (위도·경도, 도)", onInput: (v) => { ri = v; update(); } });
  pr.input.setAttribute("aria-valuetext", `${RES[ri]}도`);
  pr.input.addEventListener("input", () => pr.input.setAttribute("aria-valuetext", `${RES[ri]}도`));
  const tg = el("label", "toggle");
  tg.innerHTML = `<input type="checkbox" /> cos φ 가중`;
  tg.querySelector("input")!.addEventListener("change", (e) => { weighted = (e.target as HTMLInputElement).checked; update(); });
  const grid = el("div", "param-grid");
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  grid.append(pr.row, tg);
  const ro = readouts();
  f.panel.append(grid, ro.row);
  f.onResize(draw);
  update();
}
