// 위젯 등록부. 구현이 있는 위젯만 동적으로 불러오고, 아직 없는 위젯(status = planned)은 무엇을 하게 될지 안내하는 자리만 그린다.
import type { PageData } from "../client/main";

const IMPL: Record<string, () => Promise<{ mount: (root: HTMLElement, attrs: Record<string, string>) => void }>> = {
  "linear-map-2d": () => import("./linear-map-2d/index"),
  "relu-regions": () => import("./relu-regions/index"),
  "loss-to-estimator": () => import("./loss-to-estimator/index"),
};

export function mountWidgets(page: PageData) {
  const nodes = [...document.querySelectorAll<HTMLElement>(".widget[data-widget]")];
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      load(e.target as HTMLElement, page);
    }
  }, { rootMargin: "300px 0px" });
  nodes.forEach((n) => io.observe(n));
}

async function load(node: HTMLElement, page: PageData) {
  const name = node.dataset.widget!;
  const attrs = JSON.parse(node.dataset.attrs || "{}");
  const info = page.widgets[name];
  const impl = IMPL[name];
  if (!impl) {
    node.innerHTML = `<div class="wframe wframe-empty"><div class="wframe-head"><b></b><span class="badge">준비 중</span></div></div>`;
    node.querySelector("b")!.textContent = info?.title ?? name;
    return;
  }
  try {
    (await impl()).mount(node, attrs);
    node.dataset.ready = "true";
  } catch (err) {
    node.textContent = `위젯을 불러오지 못했다: ${String(err)}`;
  }
}
