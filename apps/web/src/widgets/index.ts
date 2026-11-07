// 위젯 등록부. 구현이 있는 위젯만 동적으로 불러오고, 아직 없는 위젯(status = planned)은 무엇을 하게 될지 안내하는 자리만 그린다.
import type { PageData } from "../client/main";

const IMPL: Record<string, () => Promise<{ mount: (root: HTMLElement, attrs: Record<string, string>) => void }>> = {
  "linear-map-2d": () => import("./linear-map-2d/index"),
  "relu-regions": () => import("./relu-regions/index"),
  "loss-to-estimator": () => import("./loss-to-estimator/index"),
  "svd-explorer": () => import("./svd-explorer/index"),
  "autodiff-graph": () => import("./autodiff-graph/index"),
  "float-precision": () => import("./float-precision/index"),
  "ode-stability": () => import("./ode-stability/index"),
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
    const mod = await impl();
    // 모든 위젯에 초기화(benchmarks.md §3.4-16): 처음 상태로 다시 그린다.
    const mountWithReset = () => {
      mod.mount(node, attrs);
      const head = node.querySelector(".wframe-head");
      if (!head) return;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "icon-btn";
      b.setAttribute("aria-label", "처음 상태로");
      b.title = "처음 상태로";
      b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>';
      b.addEventListener("click", mountWithReset);
      head.append(b);
    };
    mountWithReset();
    node.dataset.ready = "true";
  } catch (err) {
    node.textContent = `위젯을 불러오지 못했다: ${String(err)}`;
  }
}
