// 역량 레이더 (data/widgets.toml: competency-radar).
// 31.3장의 일곱 평가 질문을 축으로 한 레이더. 처음 진단(0.4장)과 지금을 겹쳐 그린다.
// 근거 규칙: ℓᵢ = min(cᵢ, eᵢ) (cᵢ 스스로 매긴 수준, eᵢ 근거로 보일 수 있는 최고 수준). 규칙을 끄면 ℓᵢ = cᵢ.
// 축(지식·기술·연구 습관)마다 평균과 최저를, 다음 목표는 ℓ가 가장 낮은 두 칸(같으면 표의 순서)을 보인다.
// 처음 값은 장의 예시 독자(first, claimed, evidenced)와 같다.
import { frame, param, presets, readouts, el, C, alpha } from "../frame";
import { pageState } from "../../client/store";

export const ITEMS = ["개념 설명", "논문 연결", "유도→코드", "실패 원인 분리", "재현 가능성", "주장 판단", "한계 명시"];
const SHORT = ["1 개념", "2 논문", "3 유도", "4 실패", "5 재현", "6 주장", "7 한계"];
export const AXIS = ["지식", "지식", "기술", "기술", "기술", "연구 습관", "연구 습관"];
const FIRST = [2, 1, 1, 1, 1, 1, 0];
const CLAIMED = [4, 3, 4, 3, 4, 3, 3];
const EVIDENCED = [4, 3, 4, 1, 4, 3, 2];

export function summarize(claimed: number[], evidenced: number[], rule: boolean) {
  const now = claimed.map((c, i) => (rule ? Math.min(c, evidenced[i]) : c));
  const axes = ["지식", "기술", "연구 습관"].map((a) => {
    const v = now.filter((_, i) => AXIS[i] === a), f = FIRST.filter((_, i) => AXIS[i] === a);
    return { a, mean: v.reduce((s, x) => s + x, 0) / v.length, min: Math.min(...v), firstMin: Math.min(...f) };
  });
  const order = now.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0] || p[1] - q[1]).map((p) => p[1]);
  const over = claimed.map((c, i) => (c > evidenced[i] ? i : -1)).filter((i) => i >= 0);
  return { now, axes, next: order.slice(0, 2), over, mean: now.reduce((s, x) => s + x, 0) / now.length };
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) <= 400;
  const compact = (root.clientWidth || 600) < 520;
  const f = frame(root, "역량 레이더", "manipulate", narrow ? 0.85 : 0.58, "일곱 평가 질문을 축으로 처음 진단과 지금의 수준을 겹쳐 그린 레이더 그림");
  const claimed = [...CLAIMED], evidenced = [...EVIDENCED];
  let rule = true, cur = 3;
  let sum = summarize(claimed, evidenced, rule);
  let labelPos: [number, number][] = [];

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2 + 6;
    const Rr = Math.min(W / 2 - (narrow ? 58 : 84), H / 2 - 40);
    const n = ITEMS.length;
    const ang = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
    const P = (i: number, v: number): [number, number] => [cx + Math.cos(ang(i)) * (v / 5) * Rr, cy + Math.sin(ang(i)) * (v / 5) * Rr];
    // 격자: 수준 1–5
    ctx.lineWidth = 1;
    for (let lv = 1; lv <= 5; lv++) {
      ctx.strokeStyle = lv === 5 ? C.mid : C.faint;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) { const p = P(i % n, lv); i ? ctx.lineTo(...p) : ctx.moveTo(...p); }
      ctx.stroke();
    }
    for (let i = 0; i < n; i++) { ctx.strokeStyle = i === cur ? C.mid : C.faint; ctx.lineWidth = i === cur ? 1.6 : 1; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(...P(i, 5)); ctx.stroke(); }
    ctx.font = "10px JetBrains Mono, monospace"; ctx.fillStyle = C.mid; ctx.textAlign = "left";
    for (let lv = 1; lv <= 5; lv++) { const p = P(0, lv); ctx.fillText(String(lv), p[0] + 4, p[1] + 3); }
    const poly = (vals: number[], stroke: string, fill: string | null, lw: number, dash: number[] = []) => {
      ctx.beginPath();
      for (let i = 0; i <= n; i++) { const p = P(i % n, vals[i % n]); i ? ctx.lineTo(...p) : ctx.moveTo(...p); }
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
    };
    poly(FIRST, C.mid, null, 1.6, [5, 4]);
    if (rule) poly(claimed, alpha(C.edge, 0.8), null, 1.4, [2, 3]);
    poly(sum.now, C.ink2, C.fill, 2.4);
    // 근거보다 높게 매긴 칸: c에서 ℓ까지의 간격
    if (rule) for (const i of sum.over) {
      ctx.strokeStyle = C.edge; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(...P(i, sum.now[i])); ctx.lineTo(...P(i, claimed[i])); ctx.stroke();
    }
    for (let i = 0; i < n; i++) {
      const target = sum.next.includes(i);
      ctx.fillStyle = target ? C.edge : C.ink2;
      ctx.beginPath(); ctx.arc(...P(i, sum.now[i]), target ? 5.5 : 3.5, 0, 7); ctx.fill();
    }
    // 축 이름
    ctx.font = `${narrow ? 600 : 600} ${narrow ? 11 : 12}px Pretendard Variable, sans-serif`;
    labelPos = [];
    for (let i = 0; i < n; i++) {
      const [x, y] = P(i, 5);
      const c = Math.cos(ang(i)), s = Math.sin(ang(i));
      const lx = x + c * 12, ly = y + s * 12 + (s > 0.3 ? 8 : s < -0.3 ? -4 : 4);
      ctx.textAlign = c > 0.25 ? "left" : c < -0.25 ? "right" : "center";
      const text = narrow ? SHORT[i] : `${i + 1}. ${ITEMS[i]}`;
      ctx.fillStyle = i === cur ? C.ink : C.mid;
      ctx.fillText(text, lx, ly);
      if (sum.next.includes(i)) {
        ctx.fillStyle = C.edge;
        ctx.fillText("다음 목표", lx, ly + 14);
      }
      labelPos.push([lx, ly]);
    }
    // 범례
    ctx.font = "11px JetBrains Mono, monospace"; ctx.textAlign = "left";
    const lg: [string, string, number[]][] = [["처음", C.mid, [5, 4]], [rule ? "지금 ℓ=min(c,e)" : "지금 ℓ=c", C.ink2, []]];
    if (rule) lg.push(["스스로 매긴 c", C.edge, [2, 3]]);
    let ly = 14;
    for (const [t, col, dash] of lg) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(8, ly - 4); ctx.lineTo(26, ly - 4); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(t, 32, ly); ly += 15;
    }
  }

  function update() {
    sum = summarize(claimed, evidenced, rule);
    for (const a of sum.axes) ro.set(a.a, `${a.a} ${a.mean.toFixed(2)} · 최저 ${a.min}`, a.min === Math.min(...sum.now) ? "hot" : "");
    ro.set("all", `전체 평균 ${sum.mean.toFixed(2)}`);
    ro.set("next", `다음 목표: ${sum.next.map((i) => ITEMS[i]).join(", ")}`, "ok");
    pageState.set("competency", { claimed: [...claimed], evidenced: [...evidenced], rule, level: sum.now, next: sum.next.map((i) => ITEMS[i]) }, root);
    draw();
  }

  const fmt = (v: number) => `${v}`;
  const pc = param({ label: "자기 c", min: 0, max: 5, step: 1, value: claimed[cur], fmt, aria: "고른 질문의 스스로 매긴 수준", onInput: (v) => { claimed[cur] = v; update(); } });
  const pe = param({ label: "근거 e", min: 0, max: 5, step: 1, value: evidenced[cur], fmt, aria: "고른 질문의 근거로 보일 수 있는 최고 수준", onInput: (v) => { evidenced[cur] = v; update(); } });
  const pick = (i: number) => { cur = i; pc.set(claimed[i]); pe.set(evidenced[i]); sel.select(String(i)); update(); };
  const sel = presets(SHORT.map((label, i) => ({ key: String(i), label: compact ? String(i + 1) : label })), (k) => pick(Number(k)), String(cur));
  sel.row.classList.add("segmented");
  sel.row.setAttribute("aria-label", "평가 질문 고르기");
  sel.row.querySelectorAll<HTMLButtonElement>(".preset").forEach((b, i) => { b.title = ITEMS[i]; b.setAttribute("aria-label", `${i + 1}. ${ITEMS[i]}`); if (compact) b.style.padding = "0 13px"; });
  for (const p of [pc, pe]) p.row.style.gridTemplateColumns = "3.4em minmax(0, 1fr) 1.2em";
  const grid = el("div", "param-grid");
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  grid.append(pc.row, pe.row);
  const tools = el("div", "wframe-row");
  const sw = el("label", "toggle");
  sw.innerHTML = `<input type="checkbox" checked /> 근거 규칙 ℓ = min(c, e)`;
  sw.querySelector("input")!.addEventListener("change", (ev) => { rule = (ev.target as HTMLInputElement).checked; update(); });
  tools.append(sw);
  const ro = readouts();
  f.panel.append(sel.row, grid, tools, ro.row);
  // 그림의 축 이름을 눌러도 질문을 고른다
  f.canvas.addEventListener("click", (e) => {
    const r = f.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, d = 40;
    labelPos.forEach(([lx, ly], i) => { const dd = Math.hypot(lx - x, ly - 4 - y); if (dd < d) { d = dd; best = i; } });
    if (best >= 0) pick(best);
  });
  f.onResize(draw);
  update();
}
