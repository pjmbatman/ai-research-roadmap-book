// 설정 계층 병합 (data/widgets.toml: config-merge, 6.3장 "조합과 override").
// 기본 설정 < 실험 파일 < 환경 변수(개인 경로) < 명령행 순서로 층을 쌓고, 키마다 최종 값과 그 값을 낸 층을 표로 보인다.
// 병합 방식 세 가지:
//   얕은 병합  {**a, **b}를 층마다(명령행은 --set마다) 되풀이 — 최상위 섹션을 통째로 바꿔 같은 섹션의 다른 키를 잃는다
//   깊은 병합  본문의 deep_merge — 바꾼 키만 바꾼다. 기본 설정에 없는 키(오타)는 조용히 새 키로 더한다
//   엄격한 깊은 병합  연습문제 strict-deep-merge처럼 기본 설정에 없는 키를 거부한다
// 층의 값은 실습(labs/research-code-engineering/configuration)의 configs/default.toml, configs/linear-baseline.toml과
// 본문의 명령행 예(train.seed=1, train.lr=0.3, 오타 train.learning_rate=0.3), 경로 절의 /scratch/me/outputs를 따른다.
import { frame, presets, readouts, el, C } from "../frame";

type Val = string | number;
type Cfg = { [section: string]: { [key: string]: Val } };
export type Layer = { key: string; label: string; short: string; cfg: Cfg[] }; // cfg: 이 층이 차례로 덮어쓰는 조각들(--set마다 하나)
export type Mode = "shallow" | "deep" | "strict";

export const DEFAULT: Cfg = {
  data: { name: "sine", n_train: 200, noise: 0.1 },
  model: { name: "poly", degree: 5 },
  train: { lr: 0.1, epochs: 2000, seed: 0 },
  paths: { output_root: "outputs" },
};
const EXP: Cfg[] = [{ model: { name: "linear" } }];
const ENV: Cfg[] = [{ paths: { output_root: "/scratch/me/outputs" } }];
const CLI: Record<string, { label: string; items: string[]; cfg: Cfg[] }> = {
  none: { label: "없음", items: [], cfg: [] },
  seedlr: { label: "train.seed=1 train.lr=0.3", items: ["train.seed=1", "train.lr=0.3"], cfg: [{ train: { seed: 1 } }, { train: { lr: 0.3 } }] },
  typo: { label: "train.learning_rate=0.3", items: ["train.learning_rate=0.3"], cfg: [{ train: { learning_rate: 0.3 } }] },
};

const clone = (c: Cfg): Cfg => JSON.parse(JSON.stringify(c));
/** 층들을 차례로 병합한다. 값마다 그 값을 마지막으로 쓴 층을 기록한다. */
export function merge(layers: Layer[], mode: Mode) {
  let cfg = clone(DEFAULT);
  const src: Record<string, string> = {};
  for (const [s, kv] of Object.entries(DEFAULT)) for (const k of Object.keys(kv)) src[`${s}.${k}`] = "default";
  let error: string | null = null;
  for (const L of layers) {
    for (const part of L.cfg) {
      for (const [s, kv] of Object.entries(part)) {
        if (mode === "strict") {
          const unknown = Object.keys(kv).filter((k) => !(DEFAULT[s] && k in DEFAULT[s]));
          if (unknown.length && !error) error = `${s}: 알 수 없는 키 ['${unknown.join("', '")}']`;
        }
        if (mode === "shallow") {
          // {**cfg, **part}: 섹션 dict를 통째로 바꾼다
          for (const k of Object.keys(cfg[s] ?? {})) delete src[`${s}.${k}`];
          cfg[s] = { ...kv };
        } else {
          cfg[s] = { ...(cfg[s] ?? {}), ...kv };
        }
        for (const k of Object.keys(kv)) src[`${s}.${k}`] = L.key;
      }
    }
  }
  if (error) cfg = {};
  return { cfg, src, error };
}
const flatKeys = (c: Cfg) => Object.entries(c).flatMap(([s, kv]) => Object.keys(kv).map((k) => `${s}.${k}`));
const get = (c: Cfg, dotted: string) => { const [s, k] = dotted.split("."); return c[s]?.[k]; };
const repr = (v: Val | undefined) => (v === undefined ? "" : typeof v === "string" ? `'${v}'` : Number.isInteger(v) ? String(v) : String(v));

export function mount(root: HTMLElement, attrs: Record<string, string>) {
  // 표 높이(행 10–11개)에 맞춰 캔버스 높이를 고른다
  const f = frame(root, "설정 계층 병합", "manipulate", (root.clientWidth < 420 ? 450 : 370) / Math.max(root.clientWidth, 300),
    "설정 키마다 기본 설정·실험 파일·환경 변수·명령행 층이 준 값과 병합된 최종 값, 그 출처를 보인 표");
  let mode: Mode = (["shallow", "deep", "strict"] as Mode[]).includes(attrs.mode as Mode) ? (attrs.mode as Mode) : "deep";
  let useExp = true, useEnv = false;
  let cli: keyof typeof CLI = "seedlr";

  const layers = (): Layer[] => [
    ...(useExp ? [{ key: "exp", label: "실험 파일", short: "exp", cfg: EXP }] : []),
    ...(useEnv ? [{ key: "env", label: "환경 변수", short: "env", cfg: ENV }] : []),
    ...(CLI[cli].cfg.length ? [{ key: "cli", label: "명령행", short: "cli", cfg: CLI[cli].cfg }] : []),
  ];
  const COLS = [
    { key: "default", label: "기본" },
    { key: "exp", label: "실험" },
    { key: "env", label: "환경" },
    { key: "cli", label: "명령행" },
  ];
  const layerCfg = (key: string): Cfg[] => key === "default" ? [DEFAULT] : key === "exp" ? (useExp ? EXP : []) : key === "env" ? (useEnv ? ENV : []) : CLI[cli].cfg;
  const layerVal = (key: string, dotted: string) => { let v: Val | undefined; for (const c of layerCfg(key)) { const x = get(c, dotted); if (x !== undefined) v = x; } return v; };

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const { cfg, src, error } = merge(layers(), mode);
    const rows = [...flatKeys(DEFAULT)];
    for (const k of flatKeys(cfg)) if (!rows.includes(k)) rows.push(k);
    if (cli === "typo" && !rows.includes("train.learning_rate")) rows.push("train.learning_rate");
    const wide = W >= 620;
    const stack = W < 420; // 좁은 화면: 키와 층 칸이 한 줄, 최종 값은 그 아래 줄
    const fs = wide ? 12.5 : 11.5;
    const rowH = Math.min(stack ? 38 : 30, (H - 40 - (error ? 30 : 0)) / (rows.length + 1));
    const keyW = Math.min(wide ? 170 : 128, W * 0.36);
    const colW = wide ? Math.min(110, (W - keyW - 220) / 4) : 34;
    const x0 = 10;
    const xs = COLS.map((_, i) => (stack ? W - 10 - (4 - i) * colW : x0 + keyW + i * colW));
    const xf = stack ? x0 + 14 : x0 + keyW + 4 * colW + 10;
    const dy = stack ? 15 : 0; // 최종 값 줄의 아래 어긋남
    const y0 = 24;
    ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
    ctx.fillStyle = C.mid;
    ctx.fillText("키", x0, y0);
    COLS.forEach((c, i) => {
      const on = c.key === "default" || layerCfg(c.key).length > 0;
      ctx.fillStyle = on ? C.ink : C.faint;
      const t = wide ? c.label : c.label.slice(0, 2);
      ctx.fillText(t, xs[i] + (wide ? 0 : (colW - ctx.measureText(t).width) / 2), y0);
    });
    ctx.fillStyle = C.ink;
    if (!stack) ctx.fillText("최종 값 (출처)", xf, y0);
    ctx.strokeStyle = C.faint;
    ctx.beginPath(); ctx.moveTo(x0, y0 + 8); ctx.lineTo(W - 10, y0 + 8); ctx.stroke();
    const fit = (s: string, w: number) => { if (ctx.measureText(s).width <= w) return s; while (s.length > 1 && ctx.measureText(s + "…").width > w) s = s.slice(0, -1); return s + "…"; };
    rows.forEach((dk, r) => {
      const y = y0 + 8 + (r + 1) * rowH - rowH * (stack ? 0.55 : 0.3);
      if (stack && r > 0) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(x0, y0 + 8 + r * rowH); ctx.lineTo(W - 10, y0 + 8 + r * rowH); ctx.stroke(); }
      const inDefault = get(DEFAULT, dk) !== undefined;
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = inDefault ? C.ink : C.edge;
      ctx.fillText(fit(dk, stack ? xs[0] - x0 - 6 : keyW - 8), x0, y);
      // 층별 칸: 이 층이 값을 주면 표시. 최종 값을 낸 층은 채운 칸, 덮어써진 층은 빈 칸
      COLS.forEach((c, i) => {
        const v = layerVal(c.key, dk);
        if (v === undefined) return;
        const won = !error && src[dk] === c.key;
        if (wide) {
          ctx.fillStyle = won ? C.edge : C.mid;
          ctx.font = `${won ? 700 : 400} ${fs}px JetBrains Mono, monospace`;
          ctx.fillText(fit(repr(v), colW - 8), xs[i], y);
          if (!won) { const w = ctx.measureText(fit(repr(v), colW - 8)).width; ctx.strokeStyle = C.mid; ctx.beginPath(); ctx.moveTo(xs[i], y - fs * 0.35); ctx.lineTo(xs[i] + w, y - fs * 0.35); ctx.stroke(); }
        } else {
          const cx = xs[i] + colW / 2, cy = y - fs * 0.35;
          ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 7);
          if (won) { ctx.fillStyle = C.edge; ctx.fill(); } else { ctx.strokeStyle = C.mid; ctx.lineWidth = 1.4; ctx.stroke(); ctx.lineWidth = 1; }
        }
      });
      // 최종 값
      ctx.font = `${fs}px JetBrains Mono, monospace`;
      if (error) { ctx.fillStyle = C.faint; ctx.fillText("—", xf, y + dy); return; }
      const v = get(cfg, dk);
      if (v === undefined) {
        ctx.fillStyle = C.edge; ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
        ctx.fillText(inDefault ? "✕ 사라짐" : "", xf, y + dy);
        return;
      }
      const tag = { default: "기본", exp: "실험", env: "환경", cli: "명령행" }[src[dk]] ?? "";
      const unused = !inDefault;
      ctx.fillStyle = unused ? C.edge : src[dk] === "default" ? C.mid : C.ink;
      ctx.font = `${src[dk] === "default" ? 400 : 700} ${fs}px JetBrains Mono, monospace`;
      const vt = fit(repr(v), W - xf - 10 - (wide ? 70 : 52));
      ctx.fillText(vt, xf, y + dy);
      ctx.font = `${fs - 1}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.mid;
      ctx.fillText(unused ? "(읽히지 않음)" : `(${tag})`, xf + ctx.measureText(vt).width + 14, y + dy);
    });
    if (error) {
      ctx.font = `700 ${fs}px JetBrains Mono, monospace`;
      ctx.fillStyle = C.edge;
      ctx.fillText(fit(`ValueError: ${error}`, W - 20), x0, H - 12);
    }
  }

  // 조작
  const modePick = presets([
    { key: "shallow", label: "얕은 병합" },
    { key: "deep", label: "깊은 병합" },
    { key: "strict", label: "엄격한 깊은 병합" },
  ], (k) => { mode = k as Mode; update(); }, mode);
  const sw = el("div", "wframe-row");
  const toggle = (label: string, on: boolean, fn: (v: boolean) => void) => {
    const t = el("label", "toggle");
    t.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} /> `;
    t.append(document.createTextNode(label));
    t.querySelector("input")!.addEventListener("change", (e) => fn((e.target as HTMLInputElement).checked));
    sw.append(t);
  };
  toggle("실험 파일", useExp, (v) => { useExp = v; update(); });
  toggle("환경 변수(개인 경로)", useEnv, (v) => { useEnv = v; update(); });
  const cliPick = presets(Object.entries(CLI).map(([key, c]) => ({ key, label: c.label })), (k) => { cli = k as keyof typeof CLI; update(); }, cli);
  const cliRow = el("div", "wframe-row");
  const cliLab = el("span", "wcap");
  cliLab.textContent = "명령행 --set";
  cliRow.append(cliLab, cliPick.row);
  cliRow.style.minWidth = "0"; // 격자 칸이 묶음의 한 줄 폭만큼 늘어나지 않게
  cliPick.row.style.minWidth = "0";
  const ro = readouts();
  modePick.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  cliPick.row.style.flexWrap = "nowrap"; // 좁은 화면에서는 묶음 안에서 가로로 넘긴다(알약 모양 유지)
  f.panel.append(modePick.row, sw, cliRow, ro.row);

  function update() {
    const { cfg, src, error } = merge(layers(), mode);
    const keys = flatKeys(DEFAULT);
    const lost = keys.filter((k) => get(cfg, k) === undefined);
    const extra = flatKeys(cfg).filter((k) => !keys.includes(k));
    const changed = keys.filter((k) => src[k] && src[k] !== "default" && get(cfg, k) !== undefined);
    if (error) ro.set("state", "실행 전에 거부", "hot");
    else if (lost.length) ro.set("state", `사라진 키 ${lost.length}개`, "hot");
    else if (extra.length) ro.set("state", `효과 없는 새 키 ${extra.length}개`, "hot");
    else ro.set("state", "모든 키 유지", "ok");
    ro.set("keys", error ? "최종 설정 없음" : `최종 키 ${flatKeys(cfg).length}개 · 바뀐 키 ${changed.length}개`);
    ro.set("lr", error ? "train.lr = —" : `train.lr = ${repr(get(cfg, "train.lr")) || "없음"}`);
    draw();
  }

  f.onResize(draw);
  update();
}
