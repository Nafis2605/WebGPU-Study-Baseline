import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { defineConfig } from "vite";

// Cross-origin isolation (COOP + COEP) gives the page a 5 µs performance.now() resolution
// (100 µs otherwise), SharedArrayBuffer for the threaded-SIMD WASM binary, and
// performance.measureUserAgentSpecificMemory(). (Port of the sibling TF.js inference study.)
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const publicDir = resolve("public");
const mimeTypes = {
  ".json": "application/json",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream",
  ".u8": "application/octet-stream",
  ".u32": "application/octet-stream",
};

// Benchmarks run against the production build (`npm run bench:serve`): no HMR client and no
// on-demand transforms in the measured page. public/ (data, weights, spec) is served in place, uncached.
function servePublicInPreview() {
  return {
    name: "serve-public-in-preview",
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = normalize(join(publicDir, decodeURIComponent(req.url.split("?")[0])));
        if (!path.startsWith(publicDir)) return next();
        let stat;
        try {
          stat = statSync(path);
        } catch {
          return next();
        }
        if (!stat.isFile()) return next();
        res.setHeader("Content-Type", mimeTypes[extname(path)] ?? "application/octet-stream");
        res.setHeader("Content-Length", stat.size);
        res.setHeader("Cache-Control", "no-store");
        for (const [k, v] of Object.entries(isolationHeaders)) res.setHeader(k, v);
        createReadStream(path).pipe(res);
      });
    },
  };
}

// Every HTML entry point that exists is built (probe and nsight pages are added as they are written).
const pages = { main: "index.html", probe: "probes/probe.html", nsight: "nsight/nsight.html" };
const input = Object.fromEntries(Object.entries(pages).filter(([, p]) => existsSync(p)).map(([k, p]) => [k, resolve(p)]));

// `vite build --mode diag` (served on :4186): unminified build for CPU profiling / kernel names only.
export default defineConfig(({ mode }) => ({
  plugins: [servePublicInPreview()],
  build: {
    copyPublicDir: false,
    rollupOptions: { input },
    ...(mode === "diag" ? { minify: false, outDir: "dist-diag" } : {}),
  },
  server: {
    host: true,
    port: 5185,
    headers: isolationHeaders,
  },
  preview: {
    port: mode === "diag" ? 4186 : 4185,
    strictPort: true,
    headers: isolationHeaders,
  },
}));
