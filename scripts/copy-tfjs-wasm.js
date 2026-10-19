// Copies the three TF.js WASM binaries into public/tfjs-wasm/ so setWasmPaths() can point at a
// same-origin location served with COOP/COEP (required for the threaded-SIMD binary).
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcDir = path.join(rootDir, "node_modules/@tensorflow/tfjs-backend-wasm/dist");
const destDir = path.join(rootDir, "public/tfjs-wasm");

const filesToCopy = [
  "tfjs-backend-wasm.wasm",
  "tfjs-backend-wasm-simd.wasm",
  "tfjs-backend-wasm-threaded-simd.wasm",
];

async function main() {
  await fs.mkdir(destDir, { recursive: true });
  for (const file of filesToCopy) {
    await fs.copyFile(path.join(srcDir, file), path.join(destDir, file));
    console.log(`Copied ${file}`);
  }
}

main().catch(err => {
  console.error(err.message);
  process.exitCode = 1;
});
