// closure 사전·사후 평가 (data/widgets.toml: closure-apriori).
// 26.3장과 같은 실험: 256칸 점성 Burgers(ν = 0.04)를 8칸 블록 필터로 걸러 32칸 ū와 subgrid 응력 τ를 만들고, 시드 0–3으로
// Smagorinsky·기울기 모델·5특징 선형 회귀를 맞춘다. 시험 시드 4–6에서 사전(a priori: τ 예측의 상대 오차)과
// 사후(a posteriori: closure를 넣은 32칸 solver의 t = 2 상대 오차)를 계산해 두 순위를 잇는다.
import { frame, presets, readouts, C, alpha } from "../frame";
import { pageState } from "../../client/store";
import { ic, run, train, aPriori, aPosteriori, filt, NF, NC, L, KEYS, NAMES, type Key } from "./model";

type Nu = "0.04" | "0.02";
const ALL: Key[] = ["none", ...KEYS];

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const f = frame(root, "closure 사전·사후 평가", "manipulate", narrow ? 1.15 : 0.56, "closure 모델별 사전(τ 예측) 오차 순위와 사후(결합 시뮬레이션) 오차 순위, 시험 시드 하나의 t = 2에서 τ 예측과 closure를 넣은 거친 격자 해");
  let model: Exclude<Key, "none"> = "lin", nuKey: Nu = "0.04";
  const fine = Array.from({ length: 7 }, (_, s) => run(ic(s, NF), L / NF, 0.04, null));
  const { models } = train(fine);
  const cache = new Map<Nu, { pri: ReturnType<typeof aPriori>; post: ReturnType<typeof aPosteriori>; refs: Float64Array[][] }>();
  const results = (k: Nu) => {
    let r = cache.get(k);
    if (!r) {
      const refs = k === "0.04" ? [4, 5, 6].map((s) => fine[s]) : [4, 5, 6].map((s) => run(ic(s, NF), L / NF, 0.02, null));
      r = { pri: aPriori(models, refs), post: aPosteriori(models, Number(k), new Map([4, 5, 6].map((s, i) => [s, refs[i]]))), refs };
      cache.set(k, r);
    }
    return r;
  };
  const priRel = (R: ReturnType<typeof results>, k: Key) => (k === "none" ? 1 : R.pri.res[k].rel); // τ = 0이면 상대 오차는 정확히 1
  const rankOf = (vals: Record<Key, number>, k: Key) => 1 + ALL.filter((o) => vals[o] < vals[k]).length;

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.font = "11px JetBrains Mono, monospace";
    const R = results(nuKey);
    const pri = Object.fromEntries(ALL.map((k) => [k, priRel(R, k)])) as Record<Key, number>;
    const post = Object.fromEntries(ALL.map((k) => [k, R.post[k].rel])) as Record<Key, number>;
    // ── 순위 잇기(slope chart)
    const sx0 = 8, sx1 = narrow ? W - 8 : Math.round(W * 0.56), sy0 = 22, sy1 = narrow ? Math.round(H * 0.47) : H - 12;
    const kor = "600 12px Pretendard Variable, sans-serif";
    const names: Record<Key, string> = { none: "모델 없음", smag: "Smagorinsky", grad: "기울기 모델", lin: "선형 회귀" };
    ctx.font = kor;
    const nameW = Math.max(...ALL.map((k) => ctx.measureText(names[k]).width));
    ctx.font = "11px JetBrains Mono, monospace";
    const ax = sx0 + nameW + 46, bx = sx1 - 44;
    const rowY = (r: number) => sy0 + 26 + ((r - 1) / 3) * (sy1 - sy0 - 40);
    ctx.fillStyle = C.mid;
    const h1 = "사전 τ", h2 = "사후 ū";
    ctx.fillText(h1, Math.max(sx0, ax - ctx.measureText(h1).width / 2), sy0);
    ctx.fillText(h2, Math.min(sx1 - ctx.measureText(h2).width, bx - ctx.measureText(h2).width / 2), sy0);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const x of [ax, bx]) { ctx.beginPath(); ctx.moveTo(x, sy0 + 12); ctx.lineTo(x, sy1); ctx.stroke(); }
    const order = [...ALL].sort((a, b) => (a === model ? 1 : b === model ? -1 : 0));
    for (const k of order) {
      const ya = rowY(rankOf(pri, k)), yb = rowY(rankOf(post, k)), sel = k === model;
      const col = sel ? C.edge : k === "none" ? alpha(C.mid, 0.6) : C.ink2;
      ctx.strokeStyle = col; ctx.lineWidth = sel ? 3 : 1.6;
      if (k === "none") ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(ax, ya); ctx.lineTo(bx, yb); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col;
      for (const [x, y] of [[ax, ya], [bx, yb]]) { ctx.beginPath(); ctx.arc(x, y, sel ? 5 : 3.5, 0, 7); ctx.fill(); }
      ctx.font = sel ? kor : "12px Pretendard Variable, sans-serif";
      ctx.fillStyle = sel ? C.edge : C.ink;
      ctx.fillText(names[k], sx0, ya - 2);
      ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillStyle = sel ? C.edge : C.mid;
      const v1 = pri[k].toFixed(3), v2 = post[k].toFixed(3);
      ctx.fillText(v1, ax - 8 - ctx.measureText(v1).width, ya + 12);
      ctx.fillText(v2, bx + 8, yb + 4);
    }
    // ── 오른쪽(좁으면 아래): 시드 4, t = 2에서 사전(τ 예측)과 사후(결합한 solver의 ū)
    const px0 = narrow ? 8 : sx1 + 22, px1 = W - 8, ly = narrow ? sy1 + 26 : 12, top = ly + 30;
    const boxes = narrow
      ? [[px0 + 4, top, W / 2 - 8, H - 20], [W / 2 + 8, top, W - 8, H - 20]]
      : [[px0 + 4, top, px1, (top + H) / 2 - 12], [px0 + 4, (top + H) / 2 + 22, px1, H - 20]];
    let lx = px0 + 4;
    for (const [nm, col, dash] of [["참값", C.ink, true], ["closure", C.edge, false]] as [string, string, boolean][]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(nm, lx + 18, ly + 4);
      lx += 28 + ctx.measureText(nm).width;
    }
    const profile = (box: number[], series: [Float64Array, string, number, number[]][], title: string, zeroBand: boolean) => {
      const [x0, y0, x1, y1] = box;
      let lo = 0, hi = 0;
      for (const [u] of series) u.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const pad = 0.08 * (hi - lo || 1); lo -= pad; hi += pad;
      const FX = (j: number) => x0 + ((j + 0.5) / NC) * (x1 - x0), FY = (v: number) => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);
      if (zeroBand && lo < 0) { ctx.fillStyle = alpha(C.mid, 0.12); ctx.fillRect(x0, FY(0), x1 - x0, y1 - FY(0)); }
      ctx.strokeStyle = C.faint; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.beginPath(); ctx.moveTo(x0, FY(0)); ctx.lineTo(x1, FY(0)); ctx.stroke();
      for (const [u, col, w, dash] of series) {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash);
        ctx.beginPath(); u.forEach((v, j) => (j ? ctx.lineTo(FX(j), FY(v)) : ctx.moveTo(FX(j), FY(v)))); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.fillStyle = C.mid; ctx.fillText(title, x0, y0 - 6);
      const s = "x"; ctx.fillText(s, x1 - ctx.measureText(s).width, y1 + 13);
    };
    const last = R.refs[0].length - 1;
    profile(boxes[0], [[R.pri.T[last], C.ink, 1.6, [5, 4]], [R.pri.res[model].pred[last], C.edge, 2.2, []]], "사전: τ(x)", true);
    const U = R.refs[0];
    profile(boxes[1], [[filt(U[last]), C.ink, 1.6, [5, 4]], [R.post[model].final[0], C.edge, 2.2, []]], "사후: ū(x)", false);
  }

  let raf = 0;
  function update() {
    const R = results(nuKey);
    const pri = Object.fromEntries(ALL.map((k) => [k, priRel(R, k)])) as Record<Key, number>;
    const post = Object.fromEntries(ALL.map((k) => [k, R.post[k].rel])) as Record<Key, number>;
    const a = R.pri.res[model];
    const r1 = rankOf(pri, model), r2 = rankOf(post, model);
    const e = R.post[model].energy;
    ro.set("rank", `오차 사전 ${a.rel.toFixed(3)} · 사후 ${R.post[model].rel.toFixed(3)} · 순위 ${r1}위 → ${r2}위`, r1 !== r2 ? "hot" : "");
    ro.set("detail", `상관 ${a.corr.toFixed(3)} · 역산란 ${Math.round(100 * a.back)}% (참 ${Math.round(100 * R.pri.trueBack)}%) · 에너지 ${e >= 0 ? "+" : "−"}${Math.abs(100 * e).toFixed(1)}%`);
    pageState.set("closure-apriori", { model, nu: Number(nuKey), apriori: a.rel, aposteriori: R.post[model].rel }, root);
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); });
  }

  const pm = presets<Exclude<Key, "none">>(KEYS.map((k) => ({ key: k, label: k === "lin" ? "선형 회귀" : k === "grad" ? "기울기" : NAMES[k] })), (k) => { model = k; update(); }, model);
  pm.row.classList.add("segmented");
  const pn = presets<Nu>([{ key: "0.04", label: "ν 0.04 학습 조건" }, { key: "0.02", label: "ν 0.02 새 Re" }], (k) => { nuKey = k; update(); }, nuKey);
  const ro = readouts();
  f.panel.append(pm.row, pn.row, ro.row);
  f.onResize(draw);
  update();
}
