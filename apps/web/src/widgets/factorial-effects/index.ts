// 2^k 요인 효과 (data/widgets.toml: factorial-effects).
// 23.3장의 가상 재현 사례: 불일치 세 개(A 셀 단위 분할, B 표준화, C 평균 손실의 λ)를 켜고 끈 8조합의 결과(제안 − baseline RMSE 차이)를
// 주효과와 A×C 교호작용으로 만든 모형 y = μ + Σ (e_j/2)·s_j + (i/2)·s_A·s_C (s = ±1)에서 계산한다.
// 추정은 본문과 같이 8조건 값의 대비(contrast)로 한다: 주효과 = D·S_j / 4, 교호작용 = D·(S_A∘S_C) / 4.
// 하나씩 바꾸기는 공개 코드 설정(− − −)에서 그 요인만 켠 조건과의 차이다. 교호작용이 있으면 이 값이 주효과와 달라진다.
// 표준화(B)의 효과는 프리셋 값으로 고정하고 조절하지 않는다(본문에서 거의 0). 막대 8개에는 들어간다.
// '본문 값'은 본문 표의 추정값(+0.146, +0.001, −0.015, −0.082)이고, 이 모형은 본문 표의 8조건을 ±0.004 안에서 다시 만든다.
import { frame, param, presets, readouts, el, C } from "../frame";

const MU = -0.0526; // 본문 표 8조건 차이의 평균
type Eff = { a: number; b: number; c: number; ac: number };
const PRESETS: Record<string, { label: string; e: Eff }> = {
  text: { label: "본문 값", e: { a: 0.146, b: 0.001, c: -0.015, ac: -0.082 } },
  none: { label: "교호작용 없음", e: { a: 0.146, b: 0.001, c: -0.015, ac: 0 } },
  only: { label: "교호작용만", e: { a: 0, b: 0, c: 0, ac: -0.1 } },
};
// 본문 표의 순서: (분할, 표준화, 평균손실) = 000, 001, 010, 011, 100, 101, 110, 111
const CFGS = Array.from({ length: 8 }, (_, i) => [(i >> 2) & 1, (i >> 1) & 1, i & 1] as const);

export function cells(e: Eff) {
  return CFGS.map(([A, B, Cc]) => {
    const sa = 2 * A - 1, sb = 2 * B - 1, sc = 2 * Cc - 1;
    return MU + (e.a / 2) * sa + (e.b / 2) * sb + (e.c / 2) * sc + (e.ac / 2) * sa * sc;
  });
}
/** 8조건 값에서 본문과 같은 대비로 효과를 추정하고, 공개 코드 설정에서 하나씩 바꾼 차이도 구한다. */
export function estimate(D: number[]) {
  const S = CFGS.map((c) => c.map((v) => 2 * v - 1));
  const dot = (g: (s: number[]) => number) => D.reduce((acc, d, i) => acc + d * g(S[i]), 0) / 4;
  return {
    main: [dot((s) => s[0]), dot((s) => s[1]), dot((s) => s[2])],
    ac: dot((s) => s[0] * s[2]),
    oat: [D[4] - D[0], D[2] - D[0], D[1] - D[0]],
  };
}

const sgn = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "2^k 요인 효과", "manipulate", narrow ? 1.5 : 0.62, "불일치 세 개를 켜고 끈 8조건의 결과 막대와, 분할에 따른 평균 손실 효과의 교호작용 그림");
  let e: Eff = { ...(PRESETS[attrs.preset] ?? PRESETS.text).e };
  let D = cells(e);

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const lo = Math.min(-0.3, ...D) - 0.02, hi = Math.max(0.15, ...D) + 0.02;
    // 왼쪽(또는 위): 8조건 가로 막대
    const bx0 = 0, bw = narrow ? W : W * 0.56, by0 = 8, bh = narrow ? H * 0.52 : H - 8;
    const labW = 74, padR = 12, top = by0 + 26, bottom = by0 + bh - 26;
    const X = (v: number) => bx0 + labW + ((v - lo) / (hi - lo)) * (bw - labW - padR);
    const rowH = (bottom - top) / 8;
    ctx.font = "600 12px Pretendard Variable, sans-serif";
    ctx.fillStyle = C.ink;
    ctx.fillText("8조건: 제안 − baseline", bx0 + 4, by0 + 12);
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillStyle = C.mid;
    ctx.fillText("분 표 평", bx0 + 6, top - 4);
    // 0 기준선과 눈금
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    for (const t of [-0.2, -0.1, 0, 0.1]) {
      if (t < lo || t > hi) continue;
      ctx.strokeStyle = t === 0 ? C.mid : C.faint;
      ctx.beginPath(); ctx.moveTo(X(t), top - 2); ctx.lineTo(X(t), bottom + 2); ctx.stroke();
      ctx.fillStyle = C.mid;
      const s = t === 0 ? "0" : t.toFixed(1);
      ctx.fillText(s, X(t) - ctx.measureText(s).width / 2, bottom + 15);
    }
    CFGS.forEach((cfg, i) => {
      const y = top + rowH * (i + 0.5);
      const isBase = i === 0, isOat = i === 1;
      ctx.fillStyle = isBase || isOat ? C.ink : C.mid;
      ctx.font = `${isBase || isOat ? "700 " : ""}12px JetBrains Mono, monospace`;
      ctx.fillText(cfg.map((v) => (v ? "O" : "-")).join("  "), bx0 + 6, y + 4);
      const h = Math.min(rowH * 0.62, 18);
      ctx.fillStyle = cfg[0] ? C.edge : C.ink2;
      const x0 = X(0), x1 = X(D[i]);
      ctx.fillRect(Math.min(x0, x1), y - h / 2, Math.max(1, Math.abs(x1 - x0)), h);
      ctx.fillStyle = C.ink;
      ctx.font = "11px JetBrains Mono, monospace";
      const s = sgn(D[i]);
      const tw = ctx.measureText(s).width;
      const tx = D[i] < 0 ? Math.min(x1, x0) - tw - 4 : Math.max(x1, x0) + 4;
      ctx.fillText(s, Math.max(bx0 + labW, Math.min(tx, bx0 + bw - tw - 2)), y + 4);
    });
    // 오른쪽(또는 아래): 교호작용 그림. x = 평균 손실 꺼짐/켜짐, 선 = 분할 방식(표준화는 평균)
    const ix0 = narrow ? 0 : W * 0.58, iw = narrow ? W : W * 0.42, iy0 = narrow ? H * 0.54 : 8, ih = narrow ? H * 0.46 - 4 : H - 8;
    const pl = ix0 + 44, pr = ix0 + iw - 14, pt = iy0 + 34, pb = iy0 + ih - 30;
    const Y = (v: number) => pb - ((v - lo) / (hi - lo)) * (pb - pt);
    const PX = [pl + (pr - pl) * 0.18, pl + (pr - pl) * 0.82];
    ctx.font = "600 12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.ink;
    ctx.fillText("분할별 평균손실 효과", ix0 + 4, iy0 + 12);
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pl, pt - 6); ctx.lineTo(pl, pb); ctx.lineTo(pr, pb); ctx.stroke();
    ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = C.mid;
    for (const t of [-0.2, -0.1, 0, 0.1]) {
      if (t < lo || t > hi) continue;
      const s = t === 0 ? "0" : t.toFixed(1);
      ctx.fillText(s, pl - 6 - ctx.measureText(s).width, Y(t) + 4);
      if (t === 0) { ctx.strokeStyle = C.faint; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(pl, Y(0)); ctx.lineTo(pr, Y(0)); ctx.stroke(); ctx.setLineDash([]); }
    }
    ctx.font = "12px Pretendard Variable, sans-serif"; ctx.fillStyle = C.mid;
    ["λ 합 손실", "λ 평균 손실"].forEach((s, k) => ctx.fillText(s, PX[k] - ctx.measureText(s).width / 2, pb + 16));
    const avg = (A: number, Cc: number) => D.filter((_, i) => CFGS[i][0] === A && CFGS[i][2] === Cc).reduce((a, b) => a + b, 0) / 2;
    const lines: [string, string, number[]][] = [
      ["무작위 분할", C.ink2, [avg(0, 0), avg(0, 1)]],
      ["셀 분할", C.edge, [avg(1, 0), avg(1, 1)]],
    ];
    const mean = [(lines[0][2][0] + lines[1][2][0]) / 2, (lines[0][2][1] + lines[1][2][1]) / 2];
    ctx.strokeStyle = C.mid; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(PX[0], Y(mean[0])); ctx.lineTo(PX[1], Y(mean[1])); ctx.stroke(); ctx.setLineDash([]);
    for (const [, col, v] of lines) {
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(PX[0], Y(v[0])); ctx.lineTo(PX[1], Y(v[1])); ctx.stroke();
      for (let k = 0; k < 2; k++) { ctx.beginPath(); ctx.arc(PX[k], Y(v[k]), 4, 0, 7); ctx.fill(); }
    }
    // 범례
    ctx.font = "11px Pretendard Variable, sans-serif";
    const leg: [string, string, boolean][] = [["무작위", C.ink2, false], ["셀 분할", C.edge, false], ["평균", C.mid, true]];
    let lx = ix0 + 4;
    const ly = iy0 + 26;
    for (const [s, col, dash] of leg) {
      ctx.strokeStyle = col; ctx.lineWidth = 2; if (dash) ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(lx, ly - 4); ctx.lineTo(lx + 14, ly - 4); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.mid; ctx.fillText(s, lx + 17, ly);
      lx += 17 + ctx.measureText(s).width + 10;
    }
  }

  function update() {
    D = cells(e);
    const est = estimate(D);
    const row = (key: string, name: string, j: number) => {
      const m = est.main[j], o = est.oat[j];
      const flip = Math.sign(o) !== Math.sign(m) && Math.abs(o - m) > 0.0005;
      ro.set(key, `${name}: 주효과 ${sgn(m)} · 하나씩 ${sgn(o)}${flip ? " (반대)" : ""}`, Math.abs(o - m) < 0.0005 ? "ok" : flip ? "hot" : "");
    };
    row("a", "분할", 0);
    row("c", "평균손실", 2);
    draw();
  }

  const fmt = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
  const sl = {
    a: param({ label: "분할", min: -0.3, max: 0.3, step: 0.001, value: e.a, fmt, aria: "셀 단위 분할의 주효과", onInput: (v) => { e.a = v; pre.select(null); update(); } }),
    c: param({ label: "평균손실", min: -0.3, max: 0.3, step: 0.001, value: e.c, fmt, aria: "평균 손실 정규화 스케일의 주효과", onInput: (v) => { e.c = v; pre.select(null); update(); } }),
    ac: param({ label: "분할×평균", min: -0.3, max: 0.3, step: 0.001, value: e.ac, fmt, aria: "분할과 평균 손실의 교호작용", onInput: (v) => { e.ac = v; pre.select(null); update(); } }),
  };
  const pre = presets(Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label })), (k) => {
    e = { ...PRESETS[k].e };
    (Object.keys(sl) as (keyof typeof sl)[]).forEach((key) => sl[key].set(e[key]));
    update();
  }, attrs.preset in PRESETS ? attrs.preset : "text");
  const grid = el("div", "param-grid");
  Object.values(sl).forEach((p) => { p.row.style.gridTemplateColumns = "5.2em minmax(0, 1fr) 3.6em"; grid.append(p.row); });
  const ro = readouts();
  f.panel.append(pre.row, grid, ro.row);
  f.onResize(draw);
  update();
}
