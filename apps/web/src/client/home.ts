// 홈 히어로의 레이어 스택: 레이어를 고르면 그 판이 떠오르고, 해당 레이어의 트랙이 카드에 나타난다.
// 아무것도 하지 않으면 레이어를 천천히 순회한다. 사용자가 조작하면 순회를 멈춘다.
interface Layer { level: number; name: string; desc: string; tracks: { key: string; code: string; name: string; n: number }[] }

export function mountStack(root: HTMLElement) {
  const layers: Layer[] = JSON.parse(root.dataset.stack || "[]");
  const base = document.body.dataset.base || "/";
  const stack = root.querySelector<HTMLElement>(".stack")!;
  const title = root.querySelector<HTMLElement>("[data-stack-title]")!;
  const desc = root.querySelector<HTMLElement>("[data-stack-desc]")!;
  const tracks = root.querySelector<HTMLElement>("[data-stack-tracks]")!;
  const buttons = [...root.querySelectorAll<HTMLButtonElement>(".stack-legend button")];
  const planes = [...root.querySelectorAll<HTMLElement>(".plane")];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let current = -1;
  let timer: number | undefined;

  const show = (level: number) => {
    current = level;
    const l = layers.find((x) => x.level === level)!;
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.level) === level)));
    planes.forEach((p) => p.classList.toggle("on", Number(p.dataset.level) === level));
    title.textContent = `L${l.level} · ${l.name}`;
    desc.textContent = l.desc;
    tracks.innerHTML = "";
    for (const t of l.tracks) {
      const a = document.createElement("a");
      a.href = `${base}tracks/${t.key}/`;
      a.className = "badge";
      a.textContent = `${t.code} ${t.name} · ${t.n}`;
      tracks.append(a);
    }
  };
  const stop = () => { clearInterval(timer); timer = undefined; stack.classList.remove("sway"); };

  buttons.forEach((b) => {
    b.addEventListener("click", () => { stop(); show(Number(b.dataset.level)); });
    b.addEventListener("mouseenter", () => { stop(); show(Number(b.dataset.level)); });
  });
  planes.forEach((p) => p.addEventListener("click", () => { stop(); show(Number(p.dataset.level)); }));

  const levels = layers.map((l) => l.level).sort((a, b) => a - b);
  show(levels[0]);
  if (!reduced) {
    stack.classList.add("sway");
    timer = window.setInterval(() => {
      const i = levels.indexOf(current);
      show(levels[(i + 1) % levels.length]);
    }, 2600);
  }
}
