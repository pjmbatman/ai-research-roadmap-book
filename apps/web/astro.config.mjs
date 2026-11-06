import { defineConfig } from "astro/config";

// BASE_PATH: GitHub Pages 프로젝트 사이트는 "/<저장소 이름>/" 아래에서 서빙된다.
export default defineConfig({
  site: process.env.SITE_URL || "https://pjmbatman.github.io",
  base: process.env.BASE_PATH || "/",
  trailingSlash: "always",
  build: { format: "directory" },
  vite: {
    worker: { format: "es" },
    server: { fs: { allow: ["../.."] } },
  },
});
