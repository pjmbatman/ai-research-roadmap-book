// tmux 세션·창·패널 (data/widgets.toml: tmux-anatomy).
// 2.2장 "작업 세션 유지"의 세 방식(그냥 실행, nohup, tmux)을 서버의 프로세스 나무로 그린다.
// SSH 연결이 끊기면 sshd가 끝나고 로그인 셸(bash)이 SIGHUP을 받는다. 셸은 끝나면서 자기 작업에 SIGHUP을 전한다.
// - 그냥 실행: 학습이 셸의 자식이라 SIGHUP으로 함께 끝난다.
// - nohup: 학습이 SIGHUP을 무시해 살아남고, 부모가 사라졌으니 PID 1에 입양된다. 그 화면으로 돌아갈 방법은 없다(출력은 nohup.out).
// - tmux: 학습은 처음부터 tmux 서버(터미널과 무관한 데몬) 아래의 패널에서 돈다. 끊기면 죽는 것은 tmux 클라이언트뿐이고, 다시 접속해 attach하면 화면이 그대로다.
// 단축키는 본문과 같은 기본 접두키 Ctrl-b 기준이다. 학습의 step 숫자는 살아 있는 동안 계속 올라간다.
import { frame, presets, readouts, el, C } from "../frame";

type Mode = "plain" | "nohup" | "tmux";
type Kind = "sys" | "conn" | "tmux" | "train" | "dead" | "pane";
interface Row { d: number; text: string; kind: Kind; sig?: boolean }

export interface State {
  mode: Mode; connected: boolean; attached: boolean; alive: boolean; adopted: boolean;
  panes: number[]; active: number; pts: number; ghosts: Row[]; event: string;
}
export const initial = (mode: Mode): State => ({ mode, connected: true, attached: mode === "tmux", alive: true, adopted: false, panes: [1], active: 0, pts: 0, ghosts: [], event: "" });

export type Action = "window" | "split" | "detach" | "drop" | "reconnect" | "attach";
export function can(s: State, a: Action): boolean {
  const t = s.mode === "tmux";
  switch (a) {
    case "window": return t && s.attached && s.panes.length < 2;
    case "split": return t && s.attached && s.panes[s.active] < 2;
    case "detach": return t && s.attached;
    case "drop": return s.connected;
    case "reconnect": return !s.connected;
    case "attach": return t && s.connected && !s.attached;
  }
}
/** 조작 하나를 적용한 다음 상태. SSH 끊김은 그 순간 SIGHUP을 받은 프로세스를 ghosts로 남긴다. */
export function step(s0: State, a: Action): State {
  const s: State = { ...s0, panes: [...s0.panes], ghosts: [], event: "" };
  if (!can(s0, a)) return s0;
  if (a === "window") { s.panes.push(1); s.active = s.panes.length - 1; s.event = "Ctrl-b c: 새 창"; }
  if (a === "split") { s.panes[s.active]++; s.event = "Ctrl-b %: 패널 좌우 분할"; }
  if (a === "detach") { s.attached = false; s.event = "Ctrl-b d: detach, 세션은 계속 돈다"; }
  if (a === "attach") { s.attached = true; s.event = "tmux attach -t exp42"; }
  if (a === "reconnect") { s.connected = true; s.pts = s0.pts + 1; s.event = "ssh gpu1: 새 로그인 셸"; }
  if (a === "drop") {
    s.connected = false;
    s.ghosts = [{ d: 1, text: `sshd: kim@pts/${s0.pts}`, kind: "dead" }, { d: 2, text: "bash (로그인 셸)", kind: "dead", sig: true }];
    if (s0.mode === "tmux" && s0.attached) s.ghosts.push({ d: 3, text: "tmux attach (클라이언트)", kind: "dead", sig: true });
    if (s0.mode === "plain" && s0.alive && s0.pts === 0) { s.ghosts.push({ d: 3, text: "python train.py", kind: "dead", sig: true }); s.alive = false; }
    if (s0.mode === "nohup" && s0.alive && !s0.adopted) s.adopted = true;
    s.attached = false;
    s.event = "SSH 끊김: 로그인 셸에 SIGHUP";
  }
  return s;
}

export function tree(s: State, k: number): Row[] {
  const rows: Row[] = [{ d: 0, text: "systemd (PID 1)", kind: "sys" }];
  const train = `python train.py · step ${k}`;
  if (s.connected) {
    rows.push({ d: 1, text: `sshd: kim@pts/${s.pts}`, kind: "conn" }, { d: 2, text: "bash (로그인 셸)", kind: "conn" });
    if (s.mode === "plain" && s.alive && s.pts === 0) rows.push({ d: 3, text: train, kind: "train" });
    if (s.mode === "nohup" && s.alive && !s.adopted) rows.push({ d: 3, text: `nohup ${train}`, kind: "train" });
    if (s.mode === "tmux" && s.attached) rows.push({ d: 3, text: "tmux attach (클라이언트)", kind: "conn" });
  }
  rows.push(...s.ghosts);
  if (s.mode === "nohup" && s.alive && s.adopted) rows.push({ d: 1, text: `${train} (SIGHUP 무시)`, kind: "train", sig: s.ghosts.length > 0 });
  if (s.mode === "tmux") {
    rows.push({ d: 1, text: "tmux 서버", kind: "tmux" }, { d: 2, text: `세션 exp42 (${s.attached ? "attached" : "detached"})`, kind: "tmux" });
    s.panes.forEach((np, w) => {
      rows.push({ d: 3, text: `창 ${w}: ${w === 0 ? "train" : "bash"}${w === s.active ? " *" : ""}`, kind: "tmux" });
      for (let p = 0; p < np; p++) {
        const isTrain = w === 0 && p === 0;
        rows.push({ d: 4, text: isTrain ? `패널 ${p}: ${train}` : `패널 ${p}: ${w === 0 ? "nvidia-smi -l 1" : "bash"}`, kind: isTrain ? "train" : "pane" });
      }
    });
  }
  return rows;
}

export function mount(root: HTMLElement) {
  const narrow = (root.clientWidth || 600) < 400;
  const f = frame(root, "tmux 세션·창·패널", "manipulate", narrow ? 1.3 : 0.5, "내 노트북의 터미널과 SSH 연결, 그리고 서버의 프로세스 나무. 학습 프로세스가 살아 있는지 보인다");
  let s = initial("tmux");
  let k = 1200;

  function screen(): string[] {
    if (!s.connected) return ["(연결 끊김)"];
    if (s.mode === "tmux" && s.attached) return [`step ${k - 1}`, `step ${k}`, "[exp42] 0:train"];
    if (s.mode === "plain" && s.alive && s.pts === 0) return [`step ${k - 1}`, `step ${k}`];
    if (s.mode === "nohup" && s.pts === 0) return ["$ nohup python train.py &", "$ ▌"];
    return ["$ ▌"];
  }

  function draw() {
    const { ctx, width: W, height: H } = f;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const rows = tree(s, k);
    ctx.textBaseline = "middle";
    // 노트북: 넓은 화면은 왼쪽, 좁은 화면은 위
    const lw = narrow ? W - 24 : Math.min(176, W * 0.25), lh = narrow ? 86 : 118;
    const lx = 12, ly = narrow ? 12 : 40;
    const sx = narrow ? 12 : lx + lw + 48, sy = narrow ? ly + lh + 34 : 16;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2;
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = C.ink; ctx.font = "700 13px Pretendard Variable, sans-serif";
    ctx.fillText("내 노트북 · 터미널", lx + 10, ly + 14);
    ctx.fillStyle = C.faint;
    ctx.fillRect(lx + 8, ly + 26, lw - 16, lh - 34);
    ctx.font = "12px JetBrains Mono, monospace";
    const scr = screen();
    scr.forEach((t, i) => { ctx.fillStyle = s.connected ? C.ink : C.mid; ctx.fillText(t, lx + 14, ly + 38 + i * 16, lw - 28); });

    // 서버 상자와 프로세스 나무
    const rowH = Math.min(26, (H - sy - 14) / Math.max(rows.length, 1));
    const ind = narrow ? 13 : 20;
    const sw = W - sx - 12, sh = rows.length * rowH + 12;
    ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2;
    ctx.strokeRect(sx, sy, sw, sh);
    ctx.fillStyle = C.mid; ctx.font = "600 12px Pretendard Variable, sans-serif";
    const sl = "서버 gpu1 · 프로세스 나무";
    ctx.fillText(sl, narrow ? sx + sw - ctx.measureText(sl).width : sx + 2, sy - 9);
    const ys = rows.map((_, i) => sy + 6 + rowH * (i + 0.5));
    // 가지: 각 행을 바로 위의 더 얕은 행(부모)에 잇는다
    ctx.lineWidth = 1;
    rows.forEach((r, i) => {
      if (r.d === 0) return;
      let p = i - 1;
      while (p >= 0 && rows[p].d >= r.d) p--;
      if (p < 0) return;
      const px = sx + 14 + rows[p].d * ind + 4, x = sx + 14 + r.d * ind;
      ctx.strokeStyle = r.kind === "dead" ? C.grid2 : C.mid;
      ctx.setLineDash(r.kind === "dead" ? [3, 3] : []);
      ctx.beginPath(); ctx.moveTo(px, ys[p] + 7); ctx.lineTo(px, ys[i]); ctx.lineTo(x - 2, ys[i]); ctx.stroke();
    });
    ctx.setLineDash([]);
    rows.forEach((r, i) => {
      const x = sx + 14 + r.d * ind, y = ys[i];
      ctx.font = `${r.kind === "train" ? "700 " : ""}12px JetBrains Mono, monospace`;
      const tw = Math.min(ctx.measureText(r.text).width, sx + sw - x - 30);
      const dot = r.kind === "train" ? C.good : r.kind === "tmux" ? C.ink2 : r.kind === "dead" ? C.grid2 : C.mid;
      ctx.fillStyle = dot;
      ctx.beginPath(); ctx.arc(x + 3, y, 3.5, 0, 7); ctx.fill();
      ctx.fillStyle = r.kind === "dead" ? C.mid : r.kind === "train" ? C.good : C.ink;
      ctx.fillText(r.text, x + 12, y, sx + sw - x - 30);
      if (r.kind === "dead") {
        ctx.strokeStyle = C.mid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x + 10, y); ctx.lineTo(x + 14 + tw, y); ctx.stroke();
      }
      if (r.sig) {
        // 방금 SIGHUP을 받은 프로세스: 코랄 번개 표시
        const bx = Math.min(x + 18 + tw, sx + sw - 16);
        ctx.strokeStyle = C.edge; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(bx + 4, y - 7); ctx.lineTo(bx, y + 1); ctx.lineTo(bx + 5, y - 1); ctx.lineTo(bx + 1, y + 7); ctx.stroke();
      }
    });

    // SSH 연결선: 노트북의 터미널 → 서버의 sshd
    const sshdRow = rows.findIndex((r) => r.text.startsWith("sshd"));
    const ax = narrow ? lx + 60 : lx + lw, ay = narrow ? ly + lh : ly + lh * 0.5;
    const bxp = narrow ? ax : sx, byp = narrow ? sy : sshdRow >= 0 ? ys[sshdRow] : sy + 40;
    ctx.strokeStyle = s.connected ? C.ink2 : C.edge; ctx.lineWidth = 2;
    ctx.setLineDash(s.connected ? [] : [6, 5]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bxp, byp); ctx.stroke();
    ctx.setLineDash([]);
    const mx = (ax + bxp) / 2, my = (ay + byp) / 2;
    ctx.fillStyle = C.bg; ctx.fillRect(mx - 26, my - 9, 52, 18);
    ctx.fillStyle = s.connected ? C.ink2 : C.edge; ctx.font = "700 12px JetBrains Mono, monospace";
    const lab = s.connected ? "SSH" : "✕ 끊김";
    ctx.fillText(lab, mx - ctx.measureText(lab).width / 2, my);
  }

  function update() {
    for (const [a, b] of btns) {
      b.disabled = !can(s, a);
      b.style.opacity = b.disabled ? "0.4" : "";
      b.style.cursor = b.disabled ? "default" : "";
      b.hidden = s.mode !== "tmux" && (a === "window" || a === "split" || a === "detach" || a === "attach");
    }
    ro.set("alive", s.alive ? "학습: 실행 중" : "학습: 종료 (SIGHUP)", s.alive ? "ok" : "hot");
    const parent = !s.alive ? "—" : s.mode === "tmux" ? "tmux 패널의 bash" : s.adopted ? "PID 1 (입양됨)" : "로그인 셸 bash";
    ro.set("parent", `부모: ${parent}`);
    const back = s.mode === "tmux" ? (s.attached ? "지금 보는 중" : "tmux attach -t exp42") : s.mode === "nohup" ? (s.adopted ? "불가, 출력은 nohup.out" : "nohup.out") : s.alive ? "지금 터미널" : "불가, 프로세스 없음";
    ro.set("back", `화면 돌아가기: ${back}`);
    if (s.event) ro.set("event", s.event); else ro.set("event", s.mode === "tmux" ? "tmux new -s exp42 에서 학습 실행" : s.mode === "nohup" ? "nohup python train.py &" : "python train.py");
    draw();
  }

  const modeBtns = presets<Mode>([{ key: "plain", label: "그냥 실행" }, { key: "nohup", label: "nohup" }, { key: "tmux", label: "tmux" }], (m) => { s = initial(m); update(); }, "tmux");
  modeBtns.row.classList.add("segmented");
  const tools = el("div", "wframe-row");
  const ACTIONS: [Action, string][] = [["window", "새 창 (c)"], ["split", "패널 분할 (%)"], ["detach", "detach (d)"], ["drop", "SSH 끊김"], ["reconnect", "다시 접속"], ["attach", "attach"]];
  const btns: [Action, HTMLButtonElement][] = ACTIONS.map(([a, label]) => {
    const b = el("button", "preset");
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", () => { s = step(s, a); update(); });
    tools.append(b);
    return [a, b];
  });
  const ro = readouts();
  f.panel.append(modeBtns.row, tools, ro.row);
  f.onResize(draw);
  update();
  // 살아 있는 학습은 step이 계속 올라간다. 위젯이 다시 그려져(초기화) 이 캔버스가 사라지면 멈춘다.
  const timer = window.setInterval(() => {
    if (!f.canvas.isConnected) { window.clearInterval(timer); return; }
    if (s.alive) { k += 1; draw(); }
  }, 700);
}
